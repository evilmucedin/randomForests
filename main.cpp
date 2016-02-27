#include <cstdlib>

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>

#include "x86intrin.h"

using namespace std;

using Features = vector<float>;

struct RandomForest {
    struct Node {
        bool isLeaf_;

        float leafValue_;

        int featureIndex_;
        float featureValue_;
        shared_ptr<Node> left_;
        shared_ptr<Node> right_;
        size_t index_;

        float eval(const Features& features) const {
            if (isLeaf_) {
                return leafValue_;
            } else {
                if (features[featureIndex_] < featureValue_) {
                    return left_->eval(features);
                } else {
                    return right_->eval(features);
                }
            }
        }

        size_t size() const {
            if (isLeaf_) {
                return 1;
            } else {
                return 1 + left_->size() + right_->size();
            }
        }
    };

    vector<shared_ptr<Node>> nodes_;

    float eval(const Features& features) const {
        float result = 0.f;
        for (const auto& node: nodes_) {
            result += node->eval(features);
        }
        return result;
    }

    size_t size() const {
        size_t size = 0;
        for (const auto& node: nodes_) {
            size += node->size();
        }
    }

    void reindex(shared_ptr<Node> node, size_t& index) {
        node->index_ = index++;
        if (!node->isLeaf_) {
            reindex(node->left_, index);
            reindex(node->right_, index);
        }
    }

    void reindex() {
        size_t index = 0;
        for (auto& node: nodes_) {
            reindex(node, index);
        }
    }
};

union Vector {
    __m256 data_;
    float floatData_[8];
};

union IVector {
    __m256i data_;
    int intData_[8];
};

struct FlatForest {
    size_t size;

    vector<int> featureIndex_;
    vector<float> featureValue_;
    vector<int> leftIndex_;
    vector<int> rightIndex_;
    IVector terminator_;

    FlatForest(const RandomForest& f) {
        size = f.size() + 1;
        terminator_.data_ = _mm256_set1_epi32(size);
    }

    Vector eval(float** features) {
        IVector current;
        current.data_ = _mm256_set1_epi32(0);
        while (_mm256_cmp_epi32_mask(current.data_, terminator_.data_, _MM_CMPINT_NEQ)) {

        }
    }
};

shared_ptr<RandomForest::Node> generateRandomNode(size_t nFeatures, size_t maxLevel, size_t level) {
    bool isLeaf = 0 == (rand() % (maxLevel - level));
    auto node = make_shared<RandomForest::Node>();
    node->isLeaf_ = isLeaf;
    if (isLeaf) {
        node->leafValue_ = static_cast<float>(rand())/RAND_MAX;
    } else {
        node->featureIndex_ = rand() % nFeatures;
        node->featureValue_ = static_cast<float>(rand())/RAND_MAX;
        node->left_ = generateRandomNode(nFeatures, maxLevel, level + 1);
        node->right_ = generateRandomNode(nFeatures, maxLevel, level + 1);
    }
    return node;
}

shared_ptr<RandomForest> generateRandomForest(size_t nFeatures, size_t nTrees, size_t nLevel) {
    auto result = make_shared<RandomForest>();
    for (size_t iTree = 0; iTree < nTrees; ++iTree) {
        result->nodes_.emplace_back(generateRandomNode(nFeatures, nLevel, 0));
    }

    return result;
}

struct ScopedTimer {
    ScopedTimer(const string& message)
        : message_(message)
        , begin_(chrono::high_resolution_clock::now())
    {
    }

    ~ScopedTimer() {
        cout << message_ << " " << chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - begin_).count() << endl;
    }

    string message_;
    chrono::high_resolution_clock::time_point begin_;
};

int main() {
    static constexpr size_t nFeatures = 100;
    shared_ptr<RandomForest> f;
    {
        ScopedTimer timer("gen");
        f = generateRandomForest(nFeatures, 1000, 3);
    }

    static constexpr size_t kN = 1000;
    vector<Features> features(kN);
    {
        ScopedTimer timer("gen features");
        for (size_t i = 0; i < kN; ++i) {
            features[i].resize(nFeatures);
            for (size_t j = 0; j < nFeatures; ++j) {
                features[i][j] = static_cast<float>(rand())/RAND_MAX;
            }
        }
    }

    {
        ScopedTimer timer("eval");
        float sum = 0;
        for (size_t j = 0; j < 30; ++j) {
            for (size_t i = 0; i < features.size(); ++i) {
                sum += f->eval(features[i]);
            }
        }
        cout << "sum: " << sum << endl;
    }

    shared_ptr<FlatForest> ff;
    {
        ScopedTimer timer("flattening");
        ff = make_shared<FlatForest>(*f);
    }

    return 0;
}
