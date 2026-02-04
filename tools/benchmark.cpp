#include "search.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <sstream>
#include <cstdlib>
#include <array>
#include <memory>

struct BenchmarkConfig {
    std::string name;
    std::string path;
    std::vector<std::string> patterns;
    std::string mode; // "exact", "fuzzy", "ripgrep", "grep"
    int max_distance;
};

struct BenchmarkResult {
    std::string config_name;
    std::string pattern;
    std::string mode;
    int num_files;
    int num_matches;
    double avg_ms;
    double min_ms;
    double max_ms;
    double std_ms;
};

std::vector<BenchmarkConfig> read_config(const std::string& filename) {
    std::vector<BenchmarkConfig> configs;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open config file '" << filename << "'\n";
        std::cerr << "Make sure the file exists and is readable.\n";
        return configs;
    }
    
    // Simple line-based config format:
    // name,path,pattern1:pattern2:pattern3,exact|fuzzy|regex,max_distance
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::stringstream ss(line);
        BenchmarkConfig config;
        
        std::string patterns_str, dist_str;
        
        std::getline(ss, config.name, ',');
        std::getline(ss, config.path, ',');
        std::getline(ss, patterns_str, ',');
        std::getline(ss, config.mode, ',');
        std::getline(ss, dist_str, ',');
        
        // Parse patterns
        std::stringstream ps(patterns_str);
        std::string pattern;
        while (std::getline(ps, pattern, ':')) {
            config.patterns.push_back(pattern);
        }
        
        config.max_distance = std::atoi(dist_str.c_str());
        
        configs.push_back(config);
    }
    
    file.close();
    return configs;
}

// Run external command and count matches
int run_external_command(const std::string& command) {
    std::array<char, 128> buffer;
    std::string result;
    int match_count = 0;
    
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        return 0;
    }
    
    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
        result += buffer.data();
        match_count++;
    }
    
    pclose(pipe);
    return match_count;
}

template<typename Func>
BenchmarkResult run_benchmark(
    const std::string& config_name,
    const std::string& pattern,
    const std::string& mode,
    const std::vector<std::string>& files,
    Func search_func,
    int num_runs = 5
) {
    std::vector<double> times;
    int matches = 0;
    
    for (int i = 0; i < num_runs; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        auto results = search_func();
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        times.push_back(duration.count() / 1000.0);
        
        if (i == 0) matches = results.size();
    }
    
    double sum = 0, min_time = times[0], max_time = times[0];
    for (double t : times) {
        sum += t;
        if (t < min_time) min_time = t;
        if (t > max_time) max_time = t;
    }
    double avg = sum / num_runs;
    
    double variance = 0;
    for (double t : times) {
        variance += (t - avg) * (t - avg);
    }
    double std_dev = std::sqrt(variance / num_runs);
    
    BenchmarkResult result;
    result.config_name = config_name;
    result.pattern = pattern;
    result.mode = mode;
    result.num_files = files.size();
    result.num_matches = matches;
    result.avg_ms = avg;
    result.min_ms = min_time;
    result.max_ms = max_time;
    result.std_ms = std_dev;
    
    return result;
}

void write_csv(const std::string& filename, const std::vector<BenchmarkResult>& results) {
    std::ofstream file(filename);
    
    file << "config,pattern,mode,files,matches,avg_ms,min_ms,max_ms,std_ms\n";
    
    for (const auto& r : results) {
        file << r.config_name << ","
             << r.pattern << ","
             << r.mode << ","
             << r.num_files << ","
             << r.num_matches << ","
             << std::fixed << std::setprecision(2)
             << r.avg_ms << ","
             << r.min_ms << ","
             << r.max_ms << ","
             << r.std_ms << "\n";
    }
    
    file.close();
    std::cout << "\nResults written to " << filename << "\n";
}

int main(int argc, char* argv[]) {
    std::string config_file = "benchmark.conf";
    std::string output_file = "benchmark.csv";
    int num_runs = 5;
    bool include_fuzzy_cpu = false;
    
    // Parse args
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_file = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "--runs" && i + 1 < argc) {
            num_runs = std::atoi(argv[++i]);
        } else if (arg == "--include-fuzzy-cpu") {
            include_fuzzy_cpu = true;
        } else if (arg == "--help") {
            std::cout << "Usage: benchmark [OPTIONS]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --config <file>       Config file (default: benchmark.conf)\n";
            std::cout << "  --output <file>       Output CSV file (default: benchmark.csv)\n";
            std::cout << "  --runs <N>            Number of runs per test (default: 5)\n";
            std::cout << "  --include-fuzzy-cpu   Include fuzzy CPU benchmarks\n";
            std::cout << "  --help                Show this help\n\n";
            std::cout << "Config format (CSV):\n";
            std::cout << "  name,path,pattern1:pattern2,exact|fuzzy|ripgrep|grep,max_distance\n\n";
            std::cout << "Example:\n";
            std::cout << "  small,/path/to/code,TODO:FIXME,exact,0\n";
            std::cout << "  large,/other/path,main:int,fuzzy,2\n";
            std::cout << "  external,/path/to/src,struct:return,ripgrep,0\n";
            return 0;
        }
    }
    
    std::cout << "Loading config from " << config_file << "\n";
    auto configs = read_config(config_file);
    std::cout << "Loaded " << configs.size() << " configurations\n";
    std::cout << "Running " << num_runs << " iterations per test\n\n";
    
    std::vector<BenchmarkResult> all_results;
    
    for (const auto& config : configs) {
        std::cout << "Config: " << config.name << " (" << config.path << ")\n";
        
        auto files = find_files(config.path);
        std::cout << "  Found " << files.size() << " files\n";
        
        for (const auto& pattern : config.patterns) {
            std::cout << "  Pattern: \"" << pattern << "\"\n";
            
            // Handle external tools (ripgrep, grep)
            if (config.mode == "ripgrep" || config.mode == "grep") {
                std::string tool = config.mode;
                std::string cmd;
                
                if (config.mode == "ripgrep") {
                    // ripgrep with file count suppression (just get matches)
                    cmd = "rg --no-heading --no-filename -c '" + pattern + "' " + config.path + " 2>/dev/null";
                } else { // grep
                    // GNU grep recursive
                    cmd = "grep -r --include='*.cpp' --include='*.cu' --include='*.h' --include='*.c' -c '" + pattern + "' " + config.path + " 2>/dev/null";
                }
                
                std::cout << "    " << tool << "... " << std::flush;
                auto result = run_benchmark(
                    config.name, pattern, tool,
                    files,
                    [&]() {
                        int count = run_external_command(cmd);
                        // Return dummy SearchResult vector
                        std::vector<SearchResult> dummy(count);
                        return dummy;
                    },
                    num_runs
                );
                all_results.push_back(result);
                std::cout << result.avg_ms << " ms\n";
                continue;
            }
            
            BenchmarkResult cpu_result;
            bool cpu_ran = true;
            
            // CPU
            if (config.mode == "fuzzy" && !include_fuzzy_cpu) {
                std::cout << "    CPU (skipped - fuzzy CPU disabled by default)\n";
                cpu_ran = false;
            } else {
                std::cout << "    CPU... " << std::flush;
                cpu_result = run_benchmark(
                    config.name, pattern, "cpu",
                    files,
                    [&]() {
                        if (config.mode == "exact") {
                            return exact_search_cpu(files, pattern, 0, false);
                        } else { // fuzzy
                            return fuzzy_search_cpu(files, pattern, config.max_distance, 0, false);
                        }
                    },
                    num_runs
                );
                all_results.push_back(cpu_result);
                std::cout << cpu_result.avg_ms << " ms\n";
            }
            
            // GPU
            std::cout << "    GPU... " << std::flush;
            auto gpu_result = run_benchmark(
                config.name, pattern, "gpu",
                files,
                [&]() {
                    if (config.mode == "exact") {
                        return exact_search_gpu(files, pattern, false);
                    } else { // fuzzy
                        return fuzzy_search_gpu(files, pattern, config.max_distance, false);
                    }
                },
                num_runs
            );
            all_results.push_back(gpu_result);
            std::cout << gpu_result.avg_ms << " ms";
            
            if (cpu_ran) {
                double speedup = cpu_result.avg_ms / gpu_result.avg_ms;
                std::cout << " (speedup: " << std::fixed << std::setprecision(2) 
                        << speedup << "x)\n";
            } else {
                std::cout << "\n";
            }
        }
        std::cout << "\n";
    }
    
    write_csv(output_file, all_results);
    
    return 0;
}
