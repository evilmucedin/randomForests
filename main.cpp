#include <stdlib.h>

#include <cstdlib>
#include <cstddef>

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>

#include "x86intrin.h"

using namespace std;

template<typename FeatureType>
struct RandomForest {
    using Features = vector<FeatureType>;

    struct Node {
        bool isLeaf_;

        FeatureType leafValue_;

        int featureIndex_;
        FeatureType featureValue_;
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

    FeatureType eval(const Features& features) const {
        FeatureType result = 0.f;
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
        return size;
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

union FloatVector {
    using AVXType = __m256;
    static const size_t kSize = 8;
    AVXType data_;
    float floatData_[8];
} __attribute__((aligned(32), packed));

union DoubleVector {
    using AVXType = __m256d;
    static const size_t kSize = 4;
    __m256d data_;
    double floatData_[4];
} __attribute__((aligned(32), packed));

union IVector8 {
    using AVXType = __m256i;
    static const size_t kSize = 8;
    __m256i data_;
    int intData_[8];
} __attribute__((aligned(32), packed));

union IVector4 {
    using AVXType = __m128i;
    static const size_t kSize = 4;
    __m128i data_;
    int intData_[4];
} __attribute__((aligned(16), packed));

template<typename T>
T InitVector(int value);

template<>
inline __m128i InitVector<__m128i>(int value) {
    return _mm_set1_epi32(value);
}

template<>
inline __m256i InitVector<__m256i>(int value) {
    return _mm256_set1_epi32(value);
}

template<typename T>
struct AVXTraits;

template<>
struct AVXTraits<float> {
    using IVectorType = IVector8;
    using FloatVectorType = FloatVector;
    static constexpr size_t kSize = 8;
};

template<>
struct AVXTraits<double> {
    using IVectorType = IVector4;
    using FloatVectorType = DoubleVector;
    static constexpr size_t kSize = 4;
};

template<typename FeatureType>
struct FlatForest {
    using RandomForestF = RandomForest<FeatureType>;
    using Traits = AVXTraits<FeatureType>;
    using IVectorType = typename Traits::IVectorType;
    using FloatVectorType = typename Traits::FloatVectorType;
    static constexpr size_t kSize = Traits::kSize;

    int iTerminator_;
    IVectorType terminator_; // should be the first field
    vector<int> featureIndex_;
    vector<FeatureType> featureValue_;
    vector<int> leftIndex_;
    vector<int> rightIndex_;
    vector<FeatureType> nodeValue_;

    FlatForest(const RandomForestF& f) {
        iTerminator_ = f.size();
        size_t size = iTerminator_ + 1;
        featureIndex_.resize(size);
        featureValue_.resize(size);
        leftIndex_.resize(size);
        rightIndex_.resize(size);
        nodeValue_.resize(size);

        for (size_t i = 1; i < f.nodes_.size(); ++i) {
            fill(f.nodes_[i - 1], f.nodes_[i]->index_);
        }
        fill(f.nodes_[f.nodes_.size() - 1], iTerminator_);
        leftIndex_[iTerminator_] = iTerminator_;
        rightIndex_[iTerminator_] = iTerminator_;
        featureIndex_[iTerminator_] = 0;
        nodeValue_[iTerminator_] = 0.f;
        featureValue_[iTerminator_] = numeric_limits<FeatureType>::max();

        size_t address = reinterpret_cast<size_t>(&(terminator_.data_));
        if (address % 16) {
            cout << "address: " << (address % 16) << " " << sizeof(terminator_.data_) << endl;
            throw std::runtime_error("bad alignment");
        }
        terminator_.data_ = InitVector<typename IVectorType::AVXType>(iTerminator_);
    }

    void fill(shared_ptr<typename RandomForestF::Node> node, int nextIndex) {
        if (node->index_ >= featureIndex_.size()) {
            throw std::runtime_error("tree invariant failed");
        }
        if (!node->isLeaf_) {
            featureIndex_[node->index_] = node->featureIndex_;
            featureValue_[node->index_] = node->featureValue_;
            leftIndex_[node->index_] = node->left_->index_;
            rightIndex_[node->index_] = node->right_->index_;
            nodeValue_[node->index_] = 0;
            fill(node->left_, nextIndex);
            fill(node->right_, nextIndex);
        } else {
            featureIndex_[node->index_] = 0;
            featureValue_[node->index_] = numeric_limits<FeatureType>::max();
            leftIndex_[node->index_] = nextIndex;
            rightIndex_[node->index_] = nextIndex;
            nodeValue_[node->index_] = node->leafValue_;
        }
    }

    FeatureType eval(const typename RandomForestF::Features& features) {
        int begin = 0;
        FeatureType result = 0.f;
        while (begin != iTerminator_) {
            result += nodeValue_[begin];
            if (features[featureIndex_[begin]] < featureValue_[begin]) {
                begin = leftIndex_[begin];
            } else {
                begin = rightIndex_[begin];
            }
        }
        return result;
    }

    static inline __m256i poorManBlend8(int mask, const __m256i& a, const __m256i& b) {
    switch (mask) {
        case 0:
            return _mm256_blend_epi32(a, b, 0);
        case 1:
                return _mm256_blend_epi32(a, b, 1);
        case 2:
                return _mm256_blend_epi32(a, b, 2);
        case 3:
                return _mm256_blend_epi32(a, b, 3);
        case 4:
                return _mm256_blend_epi32(a, b, 4);
        case 5:
                return _mm256_blend_epi32(a, b, 5);
        case 6:
                return _mm256_blend_epi32(a, b, 6);
        case 7:
                return _mm256_blend_epi32(a, b, 7);
        case 8:
                return _mm256_blend_epi32(a, b, 8);
        case 9:
                return _mm256_blend_epi32(a, b, 9);
        case 10:
                return _mm256_blend_epi32(a, b, 10);
        case 11:
                return _mm256_blend_epi32(a, b, 11);
        case 12:
                return _mm256_blend_epi32(a, b, 12);
        case 13:
                return _mm256_blend_epi32(a, b, 13);
        case 14:
                return _mm256_blend_epi32(a, b, 14);
        case 15:
                return _mm256_blend_epi32(a, b, 15);
        case 16:
                return _mm256_blend_epi32(a, b, 16);
        case 17:
                return _mm256_blend_epi32(a, b, 17);
        case 18:
                return _mm256_blend_epi32(a, b, 18);
        case 19:
                return _mm256_blend_epi32(a, b, 19);
        case 20:
                return _mm256_blend_epi32(a, b, 20);
        case 21:
                return _mm256_blend_epi32(a, b, 21);
        case 22:
                return _mm256_blend_epi32(a, b, 22);
        case 23:
                return _mm256_blend_epi32(a, b, 23);
        case 24:
                return _mm256_blend_epi32(a, b, 24);
        case 25:
                return _mm256_blend_epi32(a, b, 25);
        case 26:
                return _mm256_blend_epi32(a, b, 26);
        case 27:
                return _mm256_blend_epi32(a, b, 27);
        case 28:
                return _mm256_blend_epi32(a, b, 28);
        case 29:
                return _mm256_blend_epi32(a, b, 29);
        case 30:
                return _mm256_blend_epi32(a, b, 30);
        case 31:
                return _mm256_blend_epi32(a, b, 31);
        case 32:
                return _mm256_blend_epi32(a, b, 32);
        case 33:
                return _mm256_blend_epi32(a, b, 33);
        case 34:
                return _mm256_blend_epi32(a, b, 34);
        case 35:
                return _mm256_blend_epi32(a, b, 35);
        case 36:
                return _mm256_blend_epi32(a, b, 36);
        case 37:
                return _mm256_blend_epi32(a, b, 37);
        case 38:
                return _mm256_blend_epi32(a, b, 38);
        case 39:
                return _mm256_blend_epi32(a, b, 39);
        case 40:
                return _mm256_blend_epi32(a, b, 40);
        case 41:
                return _mm256_blend_epi32(a, b, 41);
        case 42:
                return _mm256_blend_epi32(a, b, 42);
        case 43:
                return _mm256_blend_epi32(a, b, 43);
        case 44:
                return _mm256_blend_epi32(a, b, 44);
        case 45:
                return _mm256_blend_epi32(a, b, 45);
        case 46:
                return _mm256_blend_epi32(a, b, 46);
        case 47:
                return _mm256_blend_epi32(a, b, 47);
        case 48:
                return _mm256_blend_epi32(a, b, 48);
        case 49:
                return _mm256_blend_epi32(a, b, 49);
        case 50:
                return _mm256_blend_epi32(a, b, 50);
        case 51:
                return _mm256_blend_epi32(a, b, 51);
        case 52:
                return _mm256_blend_epi32(a, b, 52);
        case 53:
                return _mm256_blend_epi32(a, b, 53);
        case 54:
                return _mm256_blend_epi32(a, b, 54);
        case 55:
                return _mm256_blend_epi32(a, b, 55);
        case 56:
                return _mm256_blend_epi32(a, b, 56);
        case 57:
                return _mm256_blend_epi32(a, b, 57);
        case 58:
                return _mm256_blend_epi32(a, b, 58);
        case 59:
                return _mm256_blend_epi32(a, b, 59);
        case 60:
                return _mm256_blend_epi32(a, b, 60);
        case 61:
                return _mm256_blend_epi32(a, b, 61);
        case 62:
                return _mm256_blend_epi32(a, b, 62);
        case 63:
                return _mm256_blend_epi32(a, b, 63);
        case 64:
                return _mm256_blend_epi32(a, b, 64);
        case 65:
                return _mm256_blend_epi32(a, b, 65);
        case 66:
                return _mm256_blend_epi32(a, b, 66);
        case 67:
                return _mm256_blend_epi32(a, b, 67);
        case 68:
                return _mm256_blend_epi32(a, b, 68);
        case 69:
                return _mm256_blend_epi32(a, b, 69);
        case 70:
                return _mm256_blend_epi32(a, b, 70);
        case 71:
                return _mm256_blend_epi32(a, b, 71);
        case 72:
                return _mm256_blend_epi32(a, b, 72);
        case 73:
                return _mm256_blend_epi32(a, b, 73);
        case 74:
                return _mm256_blend_epi32(a, b, 74);
        case 75:
                return _mm256_blend_epi32(a, b, 75);
        case 76:
                return _mm256_blend_epi32(a, b, 76);
        case 77:
                return _mm256_blend_epi32(a, b, 77);
        case 78:
                return _mm256_blend_epi32(a, b, 78);
        case 79:
                return _mm256_blend_epi32(a, b, 79);
        case 80:
                return _mm256_blend_epi32(a, b, 80);
        case 81:
                return _mm256_blend_epi32(a, b, 81);
        case 82:
                return _mm256_blend_epi32(a, b, 82);
        case 83:
                return _mm256_blend_epi32(a, b, 83);
        case 84:
                return _mm256_blend_epi32(a, b, 84);
        case 85:
                return _mm256_blend_epi32(a, b, 85);
        case 86:
                return _mm256_blend_epi32(a, b, 86);
        case 87:
                return _mm256_blend_epi32(a, b, 87);
        case 88:
                return _mm256_blend_epi32(a, b, 88);
        case 89:
                return _mm256_blend_epi32(a, b, 89);
        case 90:
                return _mm256_blend_epi32(a, b, 90);
        case 91:
                return _mm256_blend_epi32(a, b, 91);
        case 92:
                return _mm256_blend_epi32(a, b, 92);
        case 93:
                return _mm256_blend_epi32(a, b, 93);
        case 94:
                return _mm256_blend_epi32(a, b, 94);
        case 95:
                return _mm256_blend_epi32(a, b, 95);
        case 96:
                return _mm256_blend_epi32(a, b, 96);
        case 97:
                return _mm256_blend_epi32(a, b, 97);
        case 98:
                return _mm256_blend_epi32(a, b, 98);
        case 99:
                return _mm256_blend_epi32(a, b, 99);
        case 100:
                return _mm256_blend_epi32(a, b, 100);
        case 101:
                return _mm256_blend_epi32(a, b, 101);
        case 102:
                return _mm256_blend_epi32(a, b, 102);
        case 103:
                return _mm256_blend_epi32(a, b, 103);
        case 104:
                return _mm256_blend_epi32(a, b, 104);
        case 105:
                return _mm256_blend_epi32(a, b, 105);
        case 106:
                return _mm256_blend_epi32(a, b, 106);
        case 107:
                return _mm256_blend_epi32(a, b, 107);
        case 108:
                return _mm256_blend_epi32(a, b, 108);
        case 109:
                return _mm256_blend_epi32(a, b, 109);
        case 110:
                return _mm256_blend_epi32(a, b, 110);
        case 111:
                return _mm256_blend_epi32(a, b, 111);
        case 112:
                return _mm256_blend_epi32(a, b, 112);
        case 113:
                return _mm256_blend_epi32(a, b, 113);
        case 114:
                return _mm256_blend_epi32(a, b, 114);
        case 115:
                return _mm256_blend_epi32(a, b, 115);
        case 116:
                return _mm256_blend_epi32(a, b, 116);
        case 117:
                return _mm256_blend_epi32(a, b, 117);
        case 118:
                return _mm256_blend_epi32(a, b, 118);
        case 119:
                return _mm256_blend_epi32(a, b, 119);
        case 120:
                return _mm256_blend_epi32(a, b, 120);
        case 121:
                return _mm256_blend_epi32(a, b, 121);
        case 122:
                return _mm256_blend_epi32(a, b, 122);
        case 123:
                return _mm256_blend_epi32(a, b, 123);
        case 124:
                return _mm256_blend_epi32(a, b, 124);
        case 125:
                return _mm256_blend_epi32(a, b, 125);
        case 126:
                return _mm256_blend_epi32(a, b, 126);
        case 127:
                return _mm256_blend_epi32(a, b, 127);
        case 128:
                return _mm256_blend_epi32(a, b, 128);
        case 129:
                return _mm256_blend_epi32(a, b, 129);
        case 130:
                return _mm256_blend_epi32(a, b, 130);
        case 131:
                return _mm256_blend_epi32(a, b, 131);
        case 132:
                return _mm256_blend_epi32(a, b, 132);
        case 133:
                return _mm256_blend_epi32(a, b, 133);
        case 134:
                return _mm256_blend_epi32(a, b, 134);
        case 135:
                return _mm256_blend_epi32(a, b, 135);
        case 136:
                return _mm256_blend_epi32(a, b, 136);
        case 137:
                return _mm256_blend_epi32(a, b, 137);
        case 138:
                return _mm256_blend_epi32(a, b, 138);
        case 139:
                return _mm256_blend_epi32(a, b, 139);
        case 140:
                return _mm256_blend_epi32(a, b, 140);
        case 141:
                return _mm256_blend_epi32(a, b, 141);
        case 142:
                return _mm256_blend_epi32(a, b, 142);
        case 143:
                return _mm256_blend_epi32(a, b, 143);
        case 144:
                return _mm256_blend_epi32(a, b, 144);
        case 145:
                return _mm256_blend_epi32(a, b, 145);
        case 146:
                return _mm256_blend_epi32(a, b, 146);
        case 147:
                return _mm256_blend_epi32(a, b, 147);
        case 148:
                return _mm256_blend_epi32(a, b, 148);
        case 149:
                return _mm256_blend_epi32(a, b, 149);
        case 150:
                return _mm256_blend_epi32(a, b, 150);
        case 151:
                return _mm256_blend_epi32(a, b, 151);
        case 152:
                return _mm256_blend_epi32(a, b, 152);
        case 153:
                return _mm256_blend_epi32(a, b, 153);
        case 154:
                return _mm256_blend_epi32(a, b, 154);
        case 155:
                return _mm256_blend_epi32(a, b, 155);
        case 156:
                return _mm256_blend_epi32(a, b, 156);
        case 157:
                return _mm256_blend_epi32(a, b, 157);
        case 158:
                return _mm256_blend_epi32(a, b, 158);
        case 159:
                return _mm256_blend_epi32(a, b, 159);
        case 160:
                return _mm256_blend_epi32(a, b, 160);
        case 161:
                return _mm256_blend_epi32(a, b, 161);
        case 162:
                return _mm256_blend_epi32(a, b, 162);
        case 163:
                return _mm256_blend_epi32(a, b, 163);
        case 164:
                return _mm256_blend_epi32(a, b, 164);
        case 165:
                return _mm256_blend_epi32(a, b, 165);
        case 166:
                return _mm256_blend_epi32(a, b, 166);
        case 167:
                return _mm256_blend_epi32(a, b, 167);
        case 168:
                return _mm256_blend_epi32(a, b, 168);
        case 169:
                return _mm256_blend_epi32(a, b, 169);
        case 170:
                return _mm256_blend_epi32(a, b, 170);
        case 171:
                return _mm256_blend_epi32(a, b, 171);
        case 172:
                return _mm256_blend_epi32(a, b, 172);
        case 173:
                return _mm256_blend_epi32(a, b, 173);
        case 174:
                return _mm256_blend_epi32(a, b, 174);
        case 175:
                return _mm256_blend_epi32(a, b, 175);
        case 176:
                return _mm256_blend_epi32(a, b, 176);
        case 177:
                return _mm256_blend_epi32(a, b, 177);
        case 178:
                return _mm256_blend_epi32(a, b, 178);
        case 179:
                return _mm256_blend_epi32(a, b, 179);
        case 180:
                return _mm256_blend_epi32(a, b, 180);
        case 181:
                return _mm256_blend_epi32(a, b, 181);
        case 182:
                return _mm256_blend_epi32(a, b, 182);
        case 183:
                return _mm256_blend_epi32(a, b, 183);
        case 184:
                return _mm256_blend_epi32(a, b, 184);
        case 185:
                return _mm256_blend_epi32(a, b, 185);
        case 186:
                return _mm256_blend_epi32(a, b, 186);
        case 187:
                return _mm256_blend_epi32(a, b, 187);
        case 188:
                return _mm256_blend_epi32(a, b, 188);
        case 189:
                return _mm256_blend_epi32(a, b, 189);
        case 190:
                return _mm256_blend_epi32(a, b, 190);
        case 191:
                return _mm256_blend_epi32(a, b, 191);
        case 192:
                return _mm256_blend_epi32(a, b, 192);
        case 193:
                return _mm256_blend_epi32(a, b, 193);
        case 194:
                return _mm256_blend_epi32(a, b, 194);
        case 195:
                return _mm256_blend_epi32(a, b, 195);
        case 196:
                return _mm256_blend_epi32(a, b, 196);
        case 197:
                return _mm256_blend_epi32(a, b, 197);
        case 198:
                return _mm256_blend_epi32(a, b, 198);
        case 199:
                return _mm256_blend_epi32(a, b, 199);
        case 200:
                return _mm256_blend_epi32(a, b, 200);
        case 201:
                return _mm256_blend_epi32(a, b, 201);
        case 202:
                return _mm256_blend_epi32(a, b, 202);
        case 203:
                return _mm256_blend_epi32(a, b, 203);
        case 204:
                return _mm256_blend_epi32(a, b, 204);
        case 205:
                return _mm256_blend_epi32(a, b, 205);
        case 206:
                return _mm256_blend_epi32(a, b, 206);
        case 207:
                return _mm256_blend_epi32(a, b, 207);
        case 208:
                return _mm256_blend_epi32(a, b, 208);
        case 209:
                return _mm256_blend_epi32(a, b, 209);
        case 210:
                return _mm256_blend_epi32(a, b, 210);
        case 211:
                return _mm256_blend_epi32(a, b, 211);
        case 212:
                return _mm256_blend_epi32(a, b, 212);
        case 213:
                return _mm256_blend_epi32(a, b, 213);
        case 214:
                return _mm256_blend_epi32(a, b, 214);
        case 215:
                return _mm256_blend_epi32(a, b, 215);
        case 216:
                return _mm256_blend_epi32(a, b, 216);
        case 217:
                return _mm256_blend_epi32(a, b, 217);
        case 218:
                return _mm256_blend_epi32(a, b, 218);
        case 219:
                return _mm256_blend_epi32(a, b, 219);
        case 220:
                return _mm256_blend_epi32(a, b, 220);
        case 221:
                return _mm256_blend_epi32(a, b, 221);
        case 222:
                return _mm256_blend_epi32(a, b, 222);
        case 223:
                return _mm256_blend_epi32(a, b, 223);
        case 224:
                return _mm256_blend_epi32(a, b, 224);
        case 225:
                return _mm256_blend_epi32(a, b, 225);
        case 226:
                return _mm256_blend_epi32(a, b, 226);
        case 227:
                return _mm256_blend_epi32(a, b, 227);
        case 228:
                return _mm256_blend_epi32(a, b, 228);
        case 229:
                return _mm256_blend_epi32(a, b, 229);
        case 230:
                return _mm256_blend_epi32(a, b, 230);
        case 231:
                return _mm256_blend_epi32(a, b, 231);
        case 232:
                return _mm256_blend_epi32(a, b, 232);
        case 233:
                return _mm256_blend_epi32(a, b, 233);
        case 234:
                return _mm256_blend_epi32(a, b, 234);
        case 235:
                return _mm256_blend_epi32(a, b, 235);
        case 236:
                return _mm256_blend_epi32(a, b, 236);
        case 237:
                return _mm256_blend_epi32(a, b, 237);
        case 238:
                return _mm256_blend_epi32(a, b, 238);
        case 239:
                return _mm256_blend_epi32(a, b, 239);
        case 240:
                return _mm256_blend_epi32(a, b, 240);
        case 241:
                return _mm256_blend_epi32(a, b, 241);
        case 242:
                return _mm256_blend_epi32(a, b, 242);
        case 243:
                return _mm256_blend_epi32(a, b, 243);
        case 244:
                return _mm256_blend_epi32(a, b, 244);
        case 245:
                return _mm256_blend_epi32(a, b, 245);
        case 246:
                return _mm256_blend_epi32(a, b, 246);
        case 247:
                return _mm256_blend_epi32(a, b, 247);
        case 248:
                return _mm256_blend_epi32(a, b, 248);
        case 249:
                return _mm256_blend_epi32(a, b, 249);
        case 250:
                return _mm256_blend_epi32(a, b, 250);
        case 251:
                return _mm256_blend_epi32(a, b, 251);
        case 252:
                return _mm256_blend_epi32(a, b, 252);
        case 253:
                return _mm256_blend_epi32(a, b, 253);
        case 254:
                return _mm256_blend_epi32(a, b, 254);
        case 255:
                return _mm256_blend_epi32(a, b, 255);
        default:
            throw std::runtime_error("bad blend mask");
        }
    }

    FloatVector evalAVXSparse(float** features) {
        IVector8 current;
        current.data_ = _mm256_set1_epi32(0);
        FloatVector result;
        result.data_ = _mm256_set1_ps(0.f);

        FloatVector nodeValues;
        IVector8 featureIndices;
        FloatVector featureValues;
        IVector8 leftIndices;
        IVector8 rightIndices;
        FloatVector featuresHere;

        while (-1 != _mm256_movemask_epi8(_mm256_cmpeq_epi32(current.data_, terminator_.data_))) {
            nodeValues.data_ = _mm256_i32gather_ps(&nodeValue_[0], current.data_, 4);
            result.data_ = _mm256_add_ps(result.data_, nodeValues.data_);

            featureIndices.data_ = _mm256_i32gather_epi32(&featureIndex_[0], current.data_, 4);
            featureValues.data_ = _mm256_i32gather_ps(&featureValue_[0], current.data_, 4);
            leftIndices.data_ = _mm256_i32gather_epi32(&leftIndex_[0], current.data_, 4);
            rightIndices.data_ = _mm256_i32gather_epi32(&rightIndex_[0], current.data_, 4);
            for (size_t i = 0; i < 8; ++i) {
                featuresHere.floatData_[i] = features[i][featureIndices.intData_[i]];
            }
            int mask = _mm256_movemask_ps(_mm256_cmp_ps(featuresHere.data_, featureValues.data_, _CMP_LT_OS));
            current.data_ = poorManBlend8(mask, rightIndices.data_, leftIndices.data_);
        }
        return result;
    }

    FloatVector evalAVXDense(float* features0, const IVector8& offsets) {
        IVector8 current;
        current.data_ = _mm256_set1_epi32(0);
        FloatVector result;
        result.data_ = _mm256_set1_ps(0.f);

        FloatVector nodeValues;
        IVector8 featureIndices;
        FloatVector featureValues;
        IVector8 leftIndices;
        IVector8 rightIndices;
        FloatVector featuresHere;
        IVector8 featureAddresses;

        while (-1 != _mm256_movemask_epi8(_mm256_cmpeq_epi32(current.data_, terminator_.data_))) {
            nodeValues.data_ = _mm256_i32gather_ps(&nodeValue_[0], current.data_, 4);
            result.data_ = _mm256_add_ps(result.data_, nodeValues.data_);

            featureIndices.data_ = _mm256_i32gather_epi32(&featureIndex_[0], current.data_, 4);
            featureValues.data_ = _mm256_i32gather_ps(&featureValue_[0], current.data_, 4);
            leftIndices.data_ = _mm256_i32gather_epi32(&leftIndex_[0], current.data_, 4);
            rightIndices.data_ = _mm256_i32gather_epi32(&rightIndex_[0], current.data_, 4);

            featureAddresses.data_ = _mm256_add_epi32(featureIndices.data_, offsets.data_);
            featuresHere.data_ = _mm256_i32gather_ps(features0, featureAddresses.data_, 4);

            int mask = _mm256_movemask_ps(_mm256_cmp_ps(featuresHere.data_, featureValues.data_, _CMP_LT_OS));
            current.data_ = poorManBlend8(mask, rightIndices.data_, leftIndices.data_);
        }
        return result;
    }

    FloatVector evalAVX(float** features) {
        IVector8 offsets;
        for (size_t i = 0; i < 8; ++i) {
            ssize_t diff = (features[i] - features[0])/sizeof(float);
            if ( diff >= numeric_limits<int>::max() && diff <= numeric_limits<int>::min() ) {
                return evalAVXSparse(features);
            }
            offsets.intData_[i] = diff;
        }
        return evalAVXDense(features[0], offsets);
    }

    static inline __m128i poorManBlend4(int mask, const __m128i& a, const __m128i& b) {
    switch (mask) {
        case 0:
                return _mm_blend_epi32(a, b, 0);
        case 1:
                return _mm_blend_epi32(a, b, 1);
        case 2:
                return _mm_blend_epi32(a, b, 2);
        case 3:
                return _mm_blend_epi32(a, b, 3);
        case 4:
                return _mm_blend_epi32(a, b, 4);
        case 5:
                return _mm_blend_epi32(a, b, 5);
        case 6:
                return _mm_blend_epi32(a, b, 6);
        case 7:
                return _mm_blend_epi32(a, b, 7);
        case 8:
                return _mm_blend_epi32(a, b, 8);
        case 9:
                return _mm_blend_epi32(a, b, 9);
        case 10:
                return _mm_blend_epi32(a, b, 10);
        case 11:
                return _mm_blend_epi32(a, b, 11);
        case 12:
                return _mm_blend_epi32(a, b, 12);
        case 13:
                return _mm_blend_epi32(a, b, 13);
        case 14:
                return _mm_blend_epi32(a, b, 14);
        case 15:
                return _mm_blend_epi32(a, b, 15);
        default:
            throw std::runtime_error("bad blend mask");
        }
    }

    DoubleVector evalAVXSparse(double** features) {
        IVector4 current;
        current.data_ = _mm_set1_epi32(0);
        DoubleVector result;
        result.data_ = _mm256_set1_pd(0.);

        DoubleVector nodeValues;
        IVector4 featureIndices;
        DoubleVector featureValues;
        IVector4 leftIndices;
        IVector4 rightIndices;
        DoubleVector featuresHere;
        while (((1 << 16) - 1) != _mm_movemask_epi8(_mm_cmpeq_epi32(current.data_, terminator_.data_))) {
            nodeValues.data_ = _mm256_i32gather_pd(&nodeValue_[0], current.data_, 8);
            result.data_ = _mm256_add_pd(result.data_, nodeValues.data_);

            featureIndices.data_ = _mm_i32gather_epi32(&featureIndex_[0], current.data_, 4);
            featureValues.data_ = _mm256_i32gather_pd(&featureValue_[0], current.data_, 8);
            leftIndices.data_ = _mm_i32gather_epi32(&leftIndex_[0], current.data_, 4);
            rightIndices.data_ = _mm_i32gather_epi32(&rightIndex_[0], current.data_, 4);
            for (size_t i = 0; i < 4; ++i) {
                featuresHere.floatData_[i] = features[i][featureIndices.intData_[i]];
            }
            int mask = _mm256_movemask_pd(_mm256_cmp_pd(featuresHere.data_, featureValues.data_, _CMP_LT_OS));
            current.data_ = poorManBlend4(mask, rightIndices.data_, leftIndices.data_);
        }
        return std::move(result);
    }

    DoubleVector evalAVXDense(double* features0, const IVector4& offsets) {
        IVector4 current;
        current.data_ = _mm_set1_epi32(0);
        DoubleVector result;
        result.data_ = _mm256_set1_pd(0.);

        DoubleVector nodeValues;
        IVector4 featureIndices;
        DoubleVector featureValues;
        IVector4 leftIndices;
        IVector4 rightIndices;
        DoubleVector featuresHere;
        IVector4 featureAddresses;
        while (((1 << 16) - 1) != _mm_movemask_epi8(_mm_cmpeq_epi32(current.data_, terminator_.data_))) {
            nodeValues.data_ = _mm256_i32gather_pd(&nodeValue_[0], current.data_, 8);
            result.data_ = _mm256_add_pd(result.data_, nodeValues.data_);

            featureIndices.data_ = _mm_i32gather_epi32(&featureIndex_[0], current.data_, 4);
            featureValues.data_ = _mm256_i32gather_pd(&featureValue_[0], current.data_, 8);
            leftIndices.data_ = _mm_i32gather_epi32(&leftIndex_[0], current.data_, 4);
            rightIndices.data_ = _mm_i32gather_epi32(&rightIndex_[0], current.data_, 4);

            featureAddresses.data_ = _mm_add_epi32(featureIndices.data_, offsets.data_);
            featuresHere.data_ = _mm256_i32gather_pd(features0, featureAddresses.data_, 8);

            int mask = _mm256_movemask_pd(_mm256_cmp_pd(featuresHere.data_, featureValues.data_, _CMP_LT_OS));
            current.data_ = poorManBlend4(mask, rightIndices.data_, leftIndices.data_);
        }
        return result;
    }

    DoubleVector evalAVX(double** features) {
        IVector4 offsets;
        for (size_t i = 0; i < 4; ++i) {
            ssize_t diff = (features[i] - features[0])/sizeof(float);
            if ( diff >= numeric_limits<int>::max() && diff <= numeric_limits<int>::min() ) {
                return evalAVXSparse(features);
            }
            offsets.intData_[i] = diff;
        }
        return evalAVXDense(features[0], offsets);
    }

    static void* operator new(size_t size) throw()
    {
        cout << "new " << offsetof(FlatForest<FeatureType>, terminator_) << endl;
        void* mem = malloc(size + 32 + sizeof(void*));
        char* alignedMem = reinterpret_cast<char*>(mem) + sizeof(void*);
        size_t sMem = reinterpret_cast<size_t>(alignedMem);
        if (sMem % 32) {
            alignedMem += 32 - (sMem % 32);
        }
        *(reinterpret_cast<void**>(alignedMem - sizeof(void*))) = mem;

        return alignedMem;
    }

    static void operator delete(void* ptr) throw()
    {
        cout << "delete" << endl;
        void** mem = reinterpret_cast<void**>(reinterpret_cast<char*>(ptr) - sizeof(void*));

        free(*mem);
    }

    static void* operator new(std::size_t, void *) throw() = delete;
    static void operator delete (void *, void *) throw() = delete;
};

template<typename FeatureType>
shared_ptr<typename RandomForest<FeatureType>::Node> generateRandomNode(size_t nFeatures, size_t maxLevel, size_t level) {
    bool isLeaf = 0 == (rand() % (maxLevel - level));
    auto node = make_shared<typename RandomForest<FeatureType>::Node>();
    node->isLeaf_ = isLeaf;
    if (isLeaf) {
        node->leafValue_ = static_cast<FeatureType>(rand())/RAND_MAX;
    } else {
        node->featureIndex_ = rand() % nFeatures;
        node->featureValue_ = static_cast<FeatureType>(rand())/RAND_MAX;
        node->left_ = generateRandomNode<FeatureType>(nFeatures, maxLevel, level + 1);
        node->right_ = generateRandomNode<FeatureType>(nFeatures, maxLevel, level + 1);
    }
    return node;
}

template<typename FeatureType>
shared_ptr<RandomForest<FeatureType>> generateRandomForest(size_t nFeatures, size_t nTrees, size_t nLevel) {
    auto result = make_shared<RandomForest<FeatureType>>();
    for (size_t iTree = 0; iTree < nTrees; ++iTree) {
        result->nodes_.emplace_back(generateRandomNode<FeatureType>(nFeatures, nLevel, 0));
    }
    result->reindex();

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

template<typename FT>
void test() {
    cout << "================" << typeid(FT).name() << "================" << endl;

    using RF = RandomForest<FT>;
    static constexpr size_t nFeatures = 100;
    shared_ptr<RF> f;
    {
        ScopedTimer timer("gen");
        f = generateRandomForest<FT>(nFeatures, 1000, 10);
    }

    static constexpr size_t kN = 1000;
    vector<typename RF::Features> features(kN);
    {
        ScopedTimer timer("gen features");
        for (size_t i = 0; i < kN; ++i) {
            features[i].resize(nFeatures);
            for (size_t j = 0; j < nFeatures; ++j) {
                features[i][j] = static_cast<FT>(rand())/RAND_MAX;
            }
        }
    }

    {
        ScopedTimer timer("eval");
        FT sum = 0;
        for (size_t j = 0; j < 30; ++j) {
            for (size_t i = 0; i < features.size(); ++i) {
                sum += f->eval(features[i]);
            }
        }
        cout << "sum: " << sum << endl;
    }

    using FF = FlatForest<FT>;
    shared_ptr<FF> ff;
    {
        ScopedTimer timer("flattening");
        ff = shared_ptr<FF>(new FF(*f));
    }

    {
        ScopedTimer timer("flat eval");
        FT sum = 0;
        for (size_t j = 0; j < 30; ++j) {
            for (size_t i = 0; i < features.size(); ++i) {
                sum += ff->eval(features[i]);
            }
        }
        cout << "sum2: " << sum << endl;
    }

    {
        ScopedTimer timer("vector eval");
        FT sum = 0;
        for (size_t j = 0; j < 30; ++j) {
            for (size_t i = 0; i < features.size()/FF::kSize; ++i) {
                FT* data[FF::kSize];
                for (size_t k = 0; k < FF::kSize; ++k) {
                    data[k] = &features[FF::kSize*i + k][0];
                }
                typename FF::FloatVectorType v = ff->evalAVX(data);
                for (size_t k = 0; k < FF::kSize; ++k) {
                    sum += v.floatData_[k];
                }
            }
        }
        cout << "sum3: " << sum << endl;
    }
}

int main() {
    test<double>();
    test<float>();
    return 0;
}
