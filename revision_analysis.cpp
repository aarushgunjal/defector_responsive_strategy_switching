#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double BIRTH_BASELINE = 3.0;
constexpr int MAX_PRE_GENERATIONS = 1000;
constexpr std::array<int, 10> SAMPLE_GENERATIONS = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};

struct Parameters {
    int n = 100;
    double b = 5.0;
    double c = 1.0;
    double w = 1.0;
    double mu = 1.0;
    double k = 0.0;
    bool responsive = false;
    int runs = 500;
    std::uint64_t seed = 1;
};

struct TrialResult {
    bool rescued = false;
    bool censored = false;
    double allc_at_rescue = std::numeric_limits<double>::quiet_NaN();
    std::array<double, SAMPLE_GENERATIONS.size()> allc_after{};
};

struct Summary {
    double rescue_probability = 0.0;
    double rescue_se = 0.0;
    double censored_fraction = 0.0;
    double allc_at_rescue = std::numeric_limits<double>::quiet_NaN();
    double allc_at_rescue_se = std::numeric_limits<double>::quiet_NaN();
    std::array<double, SAMPLE_GENERATIONS.size()> allc_after{};
    std::array<double, SAMPLE_GENERATIONS.size()> allc_after_se{};
};

double uniform_open(std::mt19937_64& rng) {
    return std::generate_canonical<double, 64>(rng);
}

int weighted_choice(const std::array<double, 3>& weights, double total, std::mt19937_64& rng) {
    const double r = uniform_open(rng) * total;
    if (r < weights[0]) return 0;
    if (r < weights[0] + weights[1]) return 1;
    return 2;
}

TrialResult simulate_trial(const Parameters& p, std::mt19937_64& rng) {
    int nc = p.n - 1;
    int nt = 0;
    int nd = 1;
    int pre_births = 0;
    int post_births = 0;
    std::size_t next_sample = 0;
    bool post_rescue = false;

    TrialResult out;
    out.allc_after.fill(std::numeric_limits<double>::quiet_NaN());

    while (true) {
        if (!post_rescue && pre_births >= MAX_PRE_GENERATIONS * p.n) {
            out.censored = true;
            return out;
        }

        if (post_rescue && next_sample == SAMPLE_GENERATIONS.size()) {
            return out;
        }

        if (post_rescue && p.responsive && (nc == p.n || nt == p.n)) {
            const double value = static_cast<double>(nc) / p.n;
            while (next_sample < SAMPLE_GENERATIONS.size()) {
                out.allc_after[next_sample++] = value;
            }
            return out;
        }

        const double denom = static_cast<double>(p.n - 1);
        const double q = p.b - p.c;
        double pi_c = (q * (std::max(nc - 1, 0) + nt) - p.c * nd) / denom;
        if (p.responsive) pi_c -= p.k;
        const double pi_t = (q * nc + 0.5 * q * std::max(nt - 1, 0)) / denom;
        const double pi_d = p.b * nc / denom;

        const std::array<double, 3> birth_weights = {
            nc * (BIRTH_BASELINE + p.w * pi_c),
            nt * (BIRTH_BASELINE + p.w * pi_t),
            nd * (BIRTH_BASELINE + p.w * pi_d)
        };
        const double birth_total = birth_weights[0] + birth_weights[1] + birth_weights[2];
        if (birth_weights[0] < -1e-12 || birth_weights[1] < -1e-12 || birth_weights[2] < -1e-12) {
            throw std::runtime_error("Negative birth rate: increase BIRTH_BASELINE or reduce the cost sweep");
        }

        const double z = static_cast<double>(nd) / p.n;
        const double switching_intensity = p.responsive ? p.mu * z : p.mu;
        const double switch_total = switching_intensity * nc;
        const double event_total = birth_total + switch_total;
        if (!(event_total > 0.0)) throw std::runtime_error("Nonpositive event rate");

        const bool switch_event = uniform_open(rng) * event_total < switch_total;
        if (switch_event) {
            if (nc <= 0) throw std::runtime_error("Invalid switch event");
            --nc;
            ++nt;
        } else {
            const int parent = weighted_choice(birth_weights, birth_total, rng);
            const std::array<double, 3> death_weights = {
                static_cast<double>(nc), static_cast<double>(nt), static_cast<double>(nd)
            };
            const int death = weighted_choice(death_weights, static_cast<double>(p.n), rng);
            if (parent != death) {
                if (parent == 0) ++nc;
                else if (parent == 1) ++nt;
                else ++nd;
                if (death == 0) --nc;
                else if (death == 1) --nt;
                else --nd;
            }

            if (post_rescue) {
                ++post_births;
                while (next_sample < SAMPLE_GENERATIONS.size() &&
                       post_births >= SAMPLE_GENERATIONS[next_sample] * p.n) {
                    out.allc_after[next_sample] = static_cast<double>(nc) / p.n;
                    ++next_sample;
                }
            } else {
                ++pre_births;
            }
        }

        if (!post_rescue) {
            if (nd == 0) {
                out.rescued = true;
                out.allc_at_rescue = static_cast<double>(nc) / p.n;
                post_rescue = true;
                post_births = 0;
            } else if (nd == p.n) {
                return out;
            }
        }
    }
}

std::pair<double, double> mean_se(const std::vector<double>& values) {
    if (values.empty()) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        return {nan, nan};
    }
    double sum = 0.0;
    for (double value : values) sum += value;
    const double mean = sum / values.size();
    if (values.size() == 1) return {mean, std::numeric_limits<double>::quiet_NaN()};
    double ss = 0.0;
    for (double value : values) ss += (value - mean) * (value - mean);
    const double sd = std::sqrt(ss / (values.size() - 1));
    return {mean, sd / std::sqrt(static_cast<double>(values.size()))};
}

Summary run_configuration(const Parameters& p) {
    std::mt19937_64 rng(p.seed);
    int rescued = 0;
    int censored = 0;
    std::vector<double> rescue_values;
    std::array<std::vector<double>, SAMPLE_GENERATIONS.size()> after_values;
    rescue_values.reserve(p.runs);
    for (auto& values : after_values) values.reserve(p.runs);

    for (int run = 0; run < p.runs; ++run) {
        const TrialResult result = simulate_trial(p, rng);
        if (result.censored) ++censored;
        if (!result.rescued) continue;
        ++rescued;
        rescue_values.push_back(result.allc_at_rescue);
        for (std::size_t j = 0; j < SAMPLE_GENERATIONS.size(); ++j) {
            after_values[j].push_back(result.allc_after[j]);
        }
    }

    Summary out;
    out.rescue_probability = static_cast<double>(rescued) / p.runs;
    out.rescue_se = std::sqrt(out.rescue_probability * (1.0 - out.rescue_probability) / p.runs);
    out.censored_fraction = static_cast<double>(censored) / p.runs;
    std::tie(out.allc_at_rescue, out.allc_at_rescue_se) = mean_se(rescue_values);
    for (std::size_t j = 0; j < SAMPLE_GENERATIONS.size(); ++j) {
        std::tie(out.allc_after[j], out.allc_after_se[j]) = mean_se(after_values[j]);
    }
    return out;
}

void write_header(std::ofstream& out) {
    out << "scenario,mode,N,b,c,w,mu,k,runs,birth_baseline,rescue_probability,rescue_se,"
           "censored_fraction,allc_at_rescue,allc_at_rescue_se";
    for (int generation : SAMPLE_GENERATIONS) {
        out << ",allc_g" << generation << ",allc_g" << generation << "_se";
    }
    out << '\n';
}

void write_row(std::ofstream& out, const std::string& scenario, const Parameters& p, const Summary& s) {
    out << scenario << ',' << (p.responsive ? "responsive" : "constant") << ','
        << p.n << ',' << p.b << ',' << p.c << ',' << p.w << ',' << p.mu << ',' << p.k << ','
        << p.runs << ',' << BIRTH_BASELINE << ',' << s.rescue_probability << ',' << s.rescue_se << ','
        << s.censored_fraction << ',' << s.allc_at_rescue << ',' << s.allc_at_rescue_se;
    for (std::size_t j = 0; j < SAMPLE_GENERATIONS.size(); ++j) {
        out << ',' << s.allc_after[j] << ',' << s.allc_after_se[j];
    }
    out << '\n';
}

void report(const std::string& scenario, const Parameters& p, const Summary& s) {
    std::cerr << scenario << ' ' << (p.responsive ? "responsive" : "constant")
              << " N=" << p.n << " b/c=" << p.b / p.c << " w=" << p.w
              << " mu=" << p.mu << " k=" << p.k
              << " rescue=" << s.rescue_probability << " C100="
              << s.allc_after.back() << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    const std::string output_path = argc > 1 ? argv[1] : "revision_stochastic_results.csv";
    const bool quick = argc > 2 && std::string(argv[2]) == "--quick";
    std::ofstream out(output_path);
    if (!out) {
        std::cerr << "Could not open " << output_path << '\n';
        return 1;
    }
    out << std::setprecision(10);
    write_header(out);
    std::uint64_t seed = 2026101901ULL;

    const int core_runs = quick ? 80 : 500;
    const std::uint64_t zero_switch_seed = seed++;
    Summary zero_switch_summary;
    bool zero_switch_summary_ready = false;
    std::array<Summary, 2> core_mu_one_summary;
    for (bool responsive : {false, true}) {
        for (int m = 0; m <= 10; ++m) {
            Parameters p;
            p.responsive = responsive;
            p.mu = 0.2 * m;
            p.runs = core_runs;
            p.seed = (m == 0) ? zero_switch_seed : seed++;
            const Summary s = (m == 0 && zero_switch_summary_ready)
                ? zero_switch_summary
                : run_configuration(p);
            if (m == 0 && !zero_switch_summary_ready) {
                zero_switch_summary = s;
                zero_switch_summary_ready = true;
            }
            if (m == 5) core_mu_one_summary[responsive ? 1 : 0] = s;
            write_row(out, "core", p, s);
            report("core", p, s);
        }
    }

    const int robust_runs = quick ? 50 : 300;
    for (bool responsive : {false, true}) {
        for (int n : {50, 100, 200}) {
            for (double w : {0.25, 0.5, 1.0}) {
                for (double ratio : {3.0, 5.0, 7.0}) {
                    Parameters p;
                    p.responsive = responsive;
                    p.n = n;
                    p.b = ratio;
                    p.c = 1.0;
                    p.w = w;
                    p.mu = 1.0;
                    p.runs = robust_runs;
                    p.seed = seed++;
                    const Summary s = run_configuration(p);
                    write_row(out, "robustness", p, s);
                    report("robustness", p, s);
                }
            }
        }
    }

    const int cost_runs = quick ? 70 : 400;
    Parameters fixed;
    fixed.responsive = false;
    fixed.mu = 1.0;
    fixed.runs = core_runs;
    fixed.seed = seed++;
    Summary fixed_summary = core_mu_one_summary[0];
    write_row(out, "cost", fixed, fixed_summary);
    report("cost", fixed, fixed_summary);
    for (int j = 0; j <= 10; ++j) {
        Parameters p;
        p.responsive = true;
        p.mu = 1.0;
        p.k = 0.2 * j;
        p.runs = (j == 0) ? core_runs : cost_runs;
        p.seed = seed++;
        const Summary s = (j == 0) ? core_mu_one_summary[1] : run_configuration(p);
        write_row(out, "cost", p, s);
        report("cost", p, s);
    }

    return 0;
}
