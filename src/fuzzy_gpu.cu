#include "search.h"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>

struct FileInfo {
    int text_offset;
    int text_length;
    int num_lines;
    int lines_offset;
};

struct LineInfo {
    int start_pos;
    int length;
};

struct PreparedData {
    std::vector<char> all_text;
    std::vector<FileInfo> file_info;
    std::vector<std::string> filenames;
    std::vector<LineInfo> line_info;
    int total_files;
    int total_lines;
    int total_chars;
};

namespace {
    PreparedData prepare_files_for_gpu(const std::vector<std::string>& file_paths, bool verbose) {
        PreparedData data;
        data.total_files = file_paths.size();
        data.total_lines = 0;
        data.total_chars = 0;
        
        if (verbose) {
            std::cout << "Preparing " << data.total_files << " files for GPU\n";
        }
        
        for (int file_idx = 0; file_idx < data.total_files; file_idx++) {
            const std::string& filepath = file_paths[file_idx];
            
            std::ifstream file(filepath);
            if (!file.is_open()) {
                FileInfo info;
                info.text_offset = data.all_text.size();
                info.text_length = 0;
                info.num_lines = 0;
                info.lines_offset = data.line_info.size();
                data.file_info.push_back(info);
                data.filenames.push_back(filepath);
                continue;
            }
            
            std::stringstream buffer;
            buffer << file.rdbuf();
            std::string content = buffer.str();
            file.close();
            
            FileInfo info;
            info.text_offset = data.all_text.size();
            info.text_length = content.size();
            info.lines_offset = data.line_info.size();
            
            int line_start = 0;
            int line_count = 0;
            
            for (int i = 0; i < content.size(); i++) {
                if (content[i] == '\n' || i == content.size() - 1) {
                    int line_end = (content[i] == '\n') ? i : i + 1;
                    int line_length = line_end - line_start;
                    
                    LineInfo line;
                    line.start_pos = info.text_offset + line_start;
                    line.length = line_length;
                    data.line_info.push_back(line);
                    
                    line_count++;
                    line_start = i + 1;
                }
            }
            
            info.num_lines = line_count;
            data.file_info.push_back(info);
            data.filenames.push_back(filepath);
            
            data.all_text.insert(data.all_text.end(), content.begin(), content.end());
            data.total_lines += line_count;
            data.total_chars += content.size();
        }
        
        if (verbose) {
            std::cout << "Prepared: " << data.total_chars << " chars, " 
                    << data.total_lines << " lines\n";
        }
        
        return data;
    }

    __device__ int levenshtein_distance_gpu(
        const char* s1, int len1,
        const char* s2, int len2,
        int max_distance
    ) {
        if (abs(len1 - len2) > max_distance) return max_distance + 1;
        if (len1 == 0) return len2;
        if (len2 == 0) return len1;
        
        int prev_row[256];
        int curr_row[256];
        
        if (len2 >= 256) return max_distance + 1;
        
        for (int j = 0; j <= len2; j++) {
            prev_row[j] = j;
        }
        
        for (int i = 1; i <= len1; i++) {
            curr_row[0] = i;
            int min_in_row = curr_row[0];
            
            for (int j = 1; j <= len2; j++) {
                if (s1[i-1] == s2[j-1]) {
                    curr_row[j] = prev_row[j-1];
                } else {
                    int a = prev_row[j];
                    int b = curr_row[j-1];
                    int c = prev_row[j-1];
                    curr_row[j] = 1 + (a < b ? (a < c ? a : c) : (b < c ? b : c));
                }
                if (curr_row[j] < min_in_row) min_in_row = curr_row[j];
            }
            
            if (min_in_row > max_distance) {
                return max_distance + 1;
            }
            
            for (int j = 0; j <= len2; j++) {
                prev_row[j] = curr_row[j];
            }
        }
        
        return prev_row[len2];
    }

    __global__ void search_kernel_fuzzy(
        const char* all_text,
        const LineInfo* line_info,
        const char* pattern,
        int pattern_len,
        int total_lines,
        int* line_matches,
        int max_distance
    ) {
        int line_idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (line_idx >= total_lines) return;
        
        LineInfo linfo = line_info[line_idx];
        
        if (linfo.length == 0) {
            line_matches[line_idx] = 0;
            return;
        }
        
        const char* line_text = all_text + linfo.start_pos;
        int line_len = linfo.length;
        
        bool found = false;
        
        for (int window_len = pattern_len - max_distance; 
            window_len <= pattern_len + max_distance; 
            window_len++) {
            
            if (window_len <= 0 || found) continue;
            
            for (int i = 0; i <= line_len - window_len; i++) {
                int dist = levenshtein_distance_gpu(
                    pattern, pattern_len,
                    line_text + i, window_len,
                    max_distance
                );
                
                if (dist <= max_distance) {
                    found = true;
                    break;
                }
            }
        }
        
        line_matches[line_idx] = found ? 1 : 0;
    }
}

std::vector<SearchResult> fuzzy_search_gpu(
    const std::vector<std::string>& file_paths,
    const std::string& pattern,
    int max_distance,
    bool verbose
) {
    std::vector<SearchResult> results;
    
    PreparedData data = prepare_files_for_gpu(file_paths, verbose);
    
    if (data.total_files == 0 || data.total_chars == 0) {
        return results;
    }
    
    char* d_all_text;
    FileInfo* d_file_info;
    LineInfo* d_line_info;
    char* d_pattern;
    int* d_line_matches;
    
    cudaMalloc(&d_all_text, data.all_text.size());
    cudaMalloc(&d_file_info, data.file_info.size() * sizeof(FileInfo));
    cudaMalloc(&d_line_info, data.line_info.size() * sizeof(LineInfo));
    cudaMalloc(&d_pattern, pattern.size());
    cudaMalloc(&d_line_matches, data.total_lines * sizeof(int));
    
    cudaMemcpy(d_all_text, data.all_text.data(), data.all_text.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_file_info, data.file_info.data(), data.file_info.size() * sizeof(FileInfo), cudaMemcpyHostToDevice);
    cudaMemcpy(d_line_info, data.line_info.data(), data.line_info.size() * sizeof(LineInfo), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pattern, pattern.c_str(), pattern.size(), cudaMemcpyHostToDevice);
    
    int threads_per_block = 256;
    int num_blocks = (data.total_lines + threads_per_block - 1) / threads_per_block;
    
    if (verbose) {
        std::cout << "GPU fuzzy search: " << num_blocks << " blocks × " 
                << threads_per_block << " threads, max_distance=" << max_distance << "\n";
    }
    
    search_kernel_fuzzy<<<num_blocks, threads_per_block>>>(
        d_all_text,
        d_line_info,
        d_pattern,
        pattern.size(),
        data.total_lines,
        d_line_matches,
        max_distance
    );
    
    cudaDeviceSynchronize();
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << "\n";
        cudaFree(d_all_text);
        cudaFree(d_file_info);
        cudaFree(d_line_info);
        cudaFree(d_pattern);
        cudaFree(d_line_matches);
        return results;
    }
    
    std::vector<int> line_matches(data.total_lines);
    cudaMemcpy(line_matches.data(), d_line_matches, data.total_lines * sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int file_idx = 0; file_idx < data.total_files; file_idx++) {
        const FileInfo& finfo = data.file_info[file_idx];
        const std::string& filename = data.filenames[file_idx];
        
        for (int line_idx = 0; line_idx < finfo.num_lines; line_idx++) {
            int global_line_idx = finfo.lines_offset + line_idx;
            
            if (line_matches[global_line_idx] == 1) {
                LineInfo linfo = data.line_info[global_line_idx];
                
                std::string line_content(
                    data.all_text.begin() + linfo.start_pos,
                    data.all_text.begin() + linfo.start_pos + linfo.length
                );
                
                SearchResult result;
                result.filename = filename;
                result.line_number = line_idx + 1;
                result.line_content = line_content;
                
                results.push_back(result);
            }
        }
    }
    
    if (verbose) {
        std::cout << "GPU fuzzy search: " << results.size() << " matches\n";
    }
    
    cudaFree(d_all_text);
    cudaFree(d_file_info);
    cudaFree(d_line_info);
    cudaFree(d_pattern);
    cudaFree(d_line_matches);
    
    return results;
}
