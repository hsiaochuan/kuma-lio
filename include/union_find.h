#pragma once
#include <cstdint>
#include <unordered_map>

namespace faster_lio {

#include <unordered_map>
#include <cstddef>

class UnionFind {
public:
    // 查找根（路径压缩）
    size_t Find(size_t x) {
        if (parent.find(x) == parent.end()) {
            // 初始化
            parent[x] = x;
            rank[x] = 0;
            return x;
        }
        if (parent[x] != x) {
            parent[x] = Find(parent[x]); // 路径压缩
        }
        return parent[x];
    }

    // 合并
    void Union(size_t x, size_t y) {
        size_t rootX = Find(x);
        size_t rootY = Find(y);

        if (rootX == rootY) return;

        // 按秩合并
        if (rank[rootX] < rank[rootY]) {
            parent[rootX] = rootY;
        } else if (rank[rootX] > rank[rootY]) {
            parent[rootY] = rootX;
        } else {
            parent[rootY] = rootX;
            rank[rootX]++;
        }
    }

private:
    std::unordered_map<size_t, size_t> parent;
    std::unordered_map<size_t, size_t> rank;
};

}  // namespace glomap