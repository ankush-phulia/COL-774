#include <math.h>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

struct node {
    int attr_index;
    double n_entropy;
    int median;
    int maj_pred_train;
    bool isLeaf;
    node *lt_child;
    node *rt_child;
};

template <typename T>
void print(vector<T> &vec) {
    for (const auto &e : vec) {
        cout << e << ' ';
    }
    cout << endl;
}

void print(vector<pair<string, bool> > &vec) {
    for (const auto &e : vec) {
        cout << e.first.c_str() << ' ' << e.second << ' ';
        cout << endl;
    }
}

void getData(string &in_file, vector<bool> &features, vector<vector<int> > &X) {
    ifstream f_in;
    f_in.open(in_file);
    string feat;
    while (f_in >> feat && feat != "Class") {
        features.push_back((feat == "1"));
    }
    int num_feat = features.size() + 1;
    int val;
    while (f_in >> val) {
        vector<int> temp;
        temp.resize(num_feat);
        temp[0] = val;
        for (int i = 1; i < num_feat; i++) {
            f_in >> temp[i];
        }
        X.push_back(temp);
    }
}

double MI(int attr_index, vector<int> &indices, vector<vector<int> > &X) {
    vector<int> attr_values;
    int s = indices.size();
    attr_values.resize(s);
    int j = 0;
    for (int i : indices) {
        attr_values[j] = X[i][attr_index];
        j++;
    }
    // cout << attr_index << ' ' << indices.size() << endl;
    vector<int> left_indexes;
    vector<int> right_indexes;
    int median = 0;
    if (attr_index < 10) {  // continuous
        nth_element(attr_values.begin(), attr_values.begin() + (s / 2),
                    attr_values.end());
        median = attr_values[s / 2];
    } else {  // discrete
        median = 1;
    }
    int left = 0;
    int right = 0;
    // split into left and right halves
    for (int i : indices) {
        if (X[i][attr_index] < median) {
            left_indexes.push_back(i);
            left++;
        } else {
            right_indexes.push_back(i);
            right++;
        }
    }
    /*if (left == 0 || right == 0) {
            return 100.0;
    }*/
    // cout << attr_index << ' '<<median << ' '<< left << ' ' << right << endl;
    // get class p
    int class_ex[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    double probs[14] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int i : left_indexes) {
        class_ex[X[i].back() - 1]++;
    }
    for (int i : right_indexes) {
        class_ex[X[i].back() - 1 + 7]++;
    }
    // calc prob
    for (int i = 0; i < 7; i++) {
        // cout << class_ex[i] << ' ';
        probs[i] = class_ex[i] / (left + 0.0);
        // cout << probs[i] << endl;
    }
    for (int i = 7; i < 14; i++) {
        // cout << class_ex[i] << ' ';
        probs[i] = class_ex[i] / (right + 0.0);
        // cout << probs[i] << endl;
    }
    double eL = 0.0;
    double eR = 0.0;

    for (int i = 0; i < 7; i++) {
        /*if (probs[i] + probs[i+7] > 0) {
                n_entropy -= (probs[i] + probs[i+7]) * log(probs[i] +
        probs[i+7]);
        }*/
        if (probs[i] > 0) {
            eL -= probs[i] * log(probs[i]);
        }
        if (probs[i + 7] > 0) {
            eR -= probs[i + 7] * log(probs[i + 7]);
        }
    }
    double e = (eL * left + eR * right) / (right + left);
    // cout << attr_index << ' ' << e << ' ' << left << ' ' << right << endl;
    // pair<vector<int>, vector<int> > children = make_pair(left_indexes,
    // right_indexes); cout << n_entropy << endl;
    // return n_entropy;
    return e;
}

void growTree(node *nod, vector<int> &indices, uint64_t rem_atr,
              vector<vector<int> > &X) {
    uint64_t rem_attr = rem_atr;
    int classes[7] = {0, 0, 0, 0, 0, 0, 0};
    int clas = X[indices[0]].back();
    bool chk = true;
    for (int i : indices) {
        int temp = X[i].back();
        classes[temp - 1]++;
        if (temp != clas) {
            chk = false;
        }
    }
    int max_ex = 0;
    int maj = 1;
    for (int i = 0; i < 7; i++) {
        if (classes[i] > max_ex) {
            max_ex = classes[i];
            maj = i + 1;
        }
    }
    nod->maj_pred_train = maj;
    if (chk) {  // all of the same class
        nod->isLeaf = true;
        return;
    }
    if (rem_attr == 0 ||
        indices.size() == 0) {  // no attributes or examples left
        nod->isLeaf = true;
        return;
    }
    // else split
    // get lowest set bit
    int best_attr = log2(rem_attr & -rem_attr);
    double bestMI = MI(best_attr, indices, X);
    bool allsame = true;
    // get best attribute
    for (int i = best_attr + 1; i < X[0].size() - 1; i++) {
        if ((rem_attr >> i) & 1ULL) {  // if bit still 1
            // cout << i <<' ' << indices.size() << ' '<< X.size() << endl;
            double mi = MI(i, indices, X);
            if (mi < bestMI) {
                bestMI = mi;
                best_attr = i;
            }
            if (allsame && mi != bestMI) {
                allsame = false;
            }
        }
    }
    if (allsame) {  // no info gain on any attribute
        nod->isLeaf = true;
        return;
    }

    // cout << "CHOSE " << best_attr << ' ' << maj << endl;

    // split indices into left and right
    vector<int> attr_values;
    int s = indices.size();
    attr_values.resize(s);
    int j = 0;
    for (int i : indices) {
        attr_values[j] = X[i][best_attr];
        j++;
    }
    vector<int> left_indexes;
    vector<int> right_indexes;
    int median;
    if (best_attr < 10) {  // continuous
        nth_element(attr_values.begin(), attr_values.begin() + (s / 2),
                    attr_values.end());
        median = attr_values[s / 2];
    } else {  // discrete
        median = 1;
        // make attribute unavailable
        rem_attr &= ~(1ULL << best_attr);
    }
    // split into left and right halves
    for (int i : indices) {
        if (X[i][best_attr] < median) {
            left_indexes.push_back(i);
        } else {
            right_indexes.push_back(i);
        }
    }
    // write to node
    nod->attr_index = best_attr;
    nod->median = median;
    nod->n_entropy = bestMI;
    node *lt_child = new node();
    node *rt_child = new node();
    // return make_pair(left_indexes, right_indexes);
    nod->lt_child = lt_child;
    nod->rt_child = rt_child;
    // cout << bestMI << ' ';
    // cout << "left " << left_indexes.size() << ' ' << right_indexes.size() <<
    // endl;
    growTree(lt_child, left_indexes, rem_attr, X);
    // cout << "right " << right_indexes.size() << ' ' <<rt_child.n_entropy << '
    // ' << rt_child.attr_index << endl;
    growTree(rt_child, right_indexes, rem_attr, X);
}

bool predict(node *node, vector<int> &ex) {
    if (node->isLeaf) {
        return node->maj_pred_train == ex.back();
    } else {
        int attr_index = node->attr_index;
        if (ex[attr_index] < node->median) {
            return predict(node->lt_child, ex);
        } else {
            return predict(node->rt_child, ex);
        }
    }
    // return root->maj_pred_train == ex.back();
}

void test(node *root, vector<vector<int> > &X) {
    int correct = 0;
    for (auto &e : X) {
        correct += predict(root, e);
    }
    cout << correct << ' ' << X.size() << endl;
    cout << "Accuracy " << (correct * 100.0) / X.size() << endl;
}

int prune(node *node, vector<int> &indices, vector<vector<int> > &V) {
    int correct = 0;
    if (indices.size() == 0) {  // no examples
        return correct;
    }
    for (int i : indices) {  // get predictions
        if (node->maj_pred_train == V[i].back()) {
            correct++;
        }
    }
    // cout << correct << ' ';
    if (node->isLeaf) {
        return correct;
    }
    vector<int> left_indexes;
    vector<int> right_indexes;
    int median = node->median;
    int attr_index = node->attr_index;
    // cout << median << ' ' << attr_index << endl;
    // split V about this node's attribute
    for (int i : indices) {
        if (V[i][attr_index] < median) {
            left_indexes.push_back(i);
        } else {
            right_indexes.push_back(i);
        }
    }
    // if (left_indexes.size() == 0 || right_indexes.size() == 0){
    // 	return correct;
    // }
    // cout << "left " << left_indexes.size() << " right " <<
    // right_indexes.size() << endl; return correct;
    double lcorr = prune(node->lt_child, left_indexes, V);
    // cout << "done left " << left_indexes.size() << endl;
    double rcorr = prune(node->rt_child, right_indexes, V);
    // cout << "done right " << right_indexes.size() << endl;
    if (lcorr + rcorr < correct) {
        // cout << "pruned " << lcorr + rcorr - correct << endl;
        node->isLeaf = true;
        node->lt_child = NULL;
        node->rt_child = NULL;
        // delete(node->lt_child);
        // delete(node->rt_child);
        return correct;
    } else {
        return lcorr + rcorr;
    }
}

int count_node(node *node) {
    if (node->isLeaf) {
        return 1;
    } else if (node == NULL) {
        return 0;
    } else {
        return 1 + count_node(node->lt_child) + count_node(node->rt_child);
    }
}

int main(int argc, char const *argv[]) {
    string train_file = "Data/covType/train.dat";
    string valid_file = "Data/covType/valid.dat";
    string test_file = "Data/covType/test.dat";

    cout << "Getting Data" << endl;
    vector<bool> features;
    vector<vector<int> > X;
    vector<vector<int> > V;
    vector<vector<int> > T;
    getData(train_file, features, X);
    vector<bool> features1;
    getData(valid_file, features1, V);
    vector<bool> features2;
    getData(test_file, features2, T);

    cout << "Building Tree" << endl;
    node *root = new node();
    vector<int> indices;
    indices.resize(X.size());
    for (int i = 0; i < X.size(); i++) {
        indices[i] = i;
    }
    // only make num_features available, max 64
    uint64_t rem = -1;
    for (int i = features.size(); i < 64; i++) {
        rem &= ~(1ULL << i);
    }
    growTree(root, indices, rem, X);
    cout << "Nodes " << count_node(root) << endl;
    test(root, X);
    test(root, V);
    test(root, T);
    cout << "Pruning Tree" << endl;
    vector<int> indices2;
    indices2.resize(V.size());
    for (int i = 0; i < V.size(); i++) {
        indices2[i] = i;
    }
    prune(root, indices2, V);
    // root->isLeaf = true;
    cout << "Nodes " << count_node(root) << endl;
    cout << "Testing Again" << endl;
    test(root, X);
    test(root, V);
    test(root, T);
}
