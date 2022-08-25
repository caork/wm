#include <iostream>
#include "csv.h"
#include "map"
#include <cfloat>
#include "eigen3/Eigen/Dense"
#include "fstream"
#include "format.h"
using std::cout, std::endl, std::vector, std::string, std::map, std::pair;

struct wmMat {
    vector<double> x;
    double ux;
    int I;
    int A;
};

struct pairs {
    int x;
    int y;
};

struct boundary {
    double lower;
    double upper;
};

int binarySearch(double x, vector<double> &arr, int low, int high);

vector<wmMat> matReader(const string &path) {
    io::CSVReader<12> in(path);
    in.read_header(io::ignore_extra_column, "cement","slag","flyash","water","superplasticizer","coarseaggregate","fineaggregate","age","csMPa","hdbscan","fcm","hdbI");
    double x, y, z, c, v, b, n,m,ux, I, A,fcm;
    vector<wmMat> mat;
    while (in.read_row(x, y, z, c, v, b,n,m,A, I, fcm,ux)) {
        wmMat row = {{x, y, z, c, v, b,n,m}, ux, (int) fcm, (int) A};
        mat.push_back(row);
    }
    return mat;
}

vector<vector<double>> rulesReader(const string &path) {
    io::CSVReader<9> in(path);
    in.read_header(io::ignore_extra_column, "cement","slag","flyash","water","superplasticizer","coarseaggregate","fineaggregate","age","csMPa");
    double x, y, z, c, v, b,n,m,mpa;
    vector<double> xs, ys, zs, cs, vs, bs,ns,ms;
    while (in.read_row(x, y, z, c, v, b,n,m,mpa)) {
        xs.push_back(x);
        ys.push_back(y);
        zs.push_back(z);
        cs.push_back(c);
        vs.push_back(v);
        bs.push_back(b);
        ns.push_back(n);
        ms.push_back(m);
    }
    return vector<vector<double>>{xs, ys, zs, cs, vs, bs,ns,ms};
}

int fastSearch(double x, vector<vector<double>> &partition, int attribute) {
    int low = 0;
    int xRegion = binarySearch(x, partition[attribute], low, (int) partition[attribute].size() - 2);
    return xRegion;
}

int binarySearch(double x, vector<double> &arr, int low, int high) {
    if (low > high) {
        return -1;
    }
    int mid = low + (high - low) / 2;
    if (arr[mid] <= x && arr[mid + 1] > x) {
        return mid;
    } else if (arr[mid] > x) {
        return binarySearch(x, arr, low, mid - 1);
    } else {
        return binarySearch(x, arr, mid + 1, high);
    }
}

map<string, map<int, float>> wang_mendel(vector<vector<double>> &partition, vector<wmMat> &mat) {
    auto attribute = partition.size() + 3; // attribute+ux+A+I
    auto samples = mat.size();
    Eigen::ArrayXXf rules(samples, attribute - 1); // "I" is not needed
    vector<int> rule;
    for (int i = 0; i < samples; ++i) { // allocate the samples to each fuzzy region
        for (int j = 0; j < attribute - 3; ++j) {
            int index = fastSearch(mat[i].x[j], partition, j);
            rule.emplace_back(index);
        }
        rule.emplace_back(mat[i].ux);
        rule.emplace_back(mat[i].A);
        for (int j = 0; j < rule.size(); ++j) {
            rules(i, j) = (float) rule[j];
        }
        rule.clear();
    }
    map<string, map<int, float>> fuzzyRules;
    for (int i = 0; i < samples; ++i) { // convert to rule based map
        string index;
        for (int j = 0; j < attribute - 3; ++j) {
            index += '-';
            index += std::to_string((int) rules(i, j));
        }
        if (fuzzyRules.count(index)) {
            if (fuzzyRules[index].count((int) attribute - 1)) { // attributes combine to an index and A is another key
                fuzzyRules[index][(int) rules(i, (long) attribute - 2)] = rules(i, (long) attribute - 3);
            } else {
                fuzzyRules[index][(int) rules(i, (long) attribute - 2)] += rules(i, (long) attribute - 3);
            }
        } else {
            fuzzyRules[index][(int) rules(i, (long) attribute - 2)] = rules(i, (long) attribute - 3); //error
        }
    }
    for (auto &[i, theFuzzyRule]: fuzzyRules) {
        if (fuzzyRules[i].size() > 1) { // remove small weighted value
            float max = 0;
            int B; // the output value
            for (auto const &[key, value]: fuzzyRules[i]) { // find the maximum
                if (value > max) {
                    B = key;
                    max = value;
                }
            }

            fuzzyRules[i].clear();
            fuzzyRules[i][B] = max;
        }
    }

    return fuzzyRules;
}

vector<vector<double>> fuzzyRegion(vector<wmMat> &mat) {
    vector<map<int, boundary>> boundaries(11); // x ,y ,.... 6 is the dimension of dataset
    // extract boundaries
    for (int i = 0; i < boundaries.size(); i++) { // for every col
        for (auto row: mat) { // for every sample
            if (boundaries[i].find(row.I) == boundaries[i].end()) { // if dict do not include
                // insert this boundary
                boundaries[i].insert(pair<int, boundary>(row.I, boundary{row.x[i], row.x[i]}));
            } else { // if it has updated it
                if (row.x[i] > boundaries[i][row.I].upper) {
                    boundaries[i][row.I].upper = row.x[i];
                } else if (row.x[i] < boundaries[i][row.I].lower) {
                    boundaries[i][row.I].lower = row.x[i];
                }
            }
        }
    }
    vector<vector<double>> regions(8); // 6 is the number of attribute
    int index = 0;
    // stored boundaries to an array
    for (const auto &boundariesX: boundaries) {
        for (const auto &[key, val]: boundariesX) {
            regions[index].emplace_back(val.lower);
            regions[index].emplace_back(val.upper);
        }
        index++;
    }
    // remove duplicated value
    for (auto &region: regions) {
        // sort region values
        sort(region.begin(), region.end());
        double previous = -DBL_MAX;
        for (int i = 0; i < region.size(); i++) {
            if (previous != region[i]) {
                previous = region[i];
            } else {
                region.erase(region.begin() + i - 1);
                i--;
            }
        }
    }

    // alter the head and tail to minimum and maximum
    for (auto &region: regions) {
        region.insert(region.begin(), -DBL_MAX);
        region.emplace_back(DBL_MAX);
    }
    return regions;
}

void to_csv(const string &path, const vector<vector<int>> &matrix) {
    std::ofstream out(path);

    for (auto &row: matrix) {
        for (auto i = 0; i < row.size(); ++i) {
            if (i == row.size() - 1) {
                out << row[i];
            } else {
                out << row[i] << ',';
            }
        }
        out << '\n';
    }
}

vector<vector<int>>
predict(map<string, map<int, float>> &model, vector<wmMat> &test, vector<vector<double>> &partition) {
    vector<vector<int>> result;
    vector<int> p; // storage predict and true value
    int numberOfRow = 0;
    for (auto &row: test) {
        string index;
        p.emplace_back(row.A);
        for (double i: row.x) {
            for (int j = 0; j < partition[numberOfRow].size(); ++j) {
                if (i <= partition[numberOfRow][j]) {
                    index += '-';
                    index += std::to_string((int) j);
                    break;
                }
            }
            numberOfRow++;
        }
        numberOfRow = 0;
        if (model.count(index)) {
            for (const auto &[key, value]: model[index]) {
                p.emplace_back(key);
                break;
            }
        } else {
            cout << index << endl;
        }
        result.emplace_back(p);
        p.clear();
    }
    return result;
}

void fuzzyMatlab(vector<vector<double>> &partition,map<string,map<int,float>> &model){
    string fuzzyLogicCode = "fis = mamfis('Name',\"HDBSCANFS\");\n";
    int attributeName = 1000;
    for (const auto& attribute : partition) {
        string input = SFormat("fis = addInput(fis,[{0} {1}],'Name',\"{2}\");\n",0,attribute[attribute.size()-2],attributeName);
        fuzzyLogicCode += input; // add attribute to code;
        char regionName = 67; // start from a
        for (int i = 0; i < attribute.size()-1; ++i) {
            fuzzyLogicCode += SFormat("fis = addMF(fis,\"{0}\",\"{1}\",[{2} {3}],'Name',\"{4}\");\n",attributeName,"gaussmf",attribute[i],attribute[i+1],regionName); // addMF
            regionName++; // update
        }
        attributeName++; // update
    }
    cout<<fuzzyLogicCode;
}

void modelClean(map<string,map<int,float>> &model){
    vector<string> useless;
    for (const auto& rule:model){
        for (const auto& key:rule.second){
            if (key.first<0){
                useless.emplace_back(rule.first);
            }
        }
    }
    for (const auto& key:useless) {
        model.erase(key);
    }
}

int main() {
    auto mat = matReader("concreteResult.csv");
    // auto partition = rulesReader("wmrules.csv");
    auto partition = fuzzyRegion(mat);
    auto model = wang_mendel(partition, mat);
    modelClean(model);
    fuzzyMatlab(partition,model);
    for(const auto& m:model){
        cout<<m.first<<endl;
    }
    for (auto i=0;i<20;i++){
        cout<<"---------------------"<<endl;
    }

    auto test = matReader("concreteResult.csv");
    auto result = predict(model, test, partition);
    to_csv("results.csv", result);
}