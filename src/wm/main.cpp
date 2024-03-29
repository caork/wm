#include <iostream>
#include "../tool/csv.h"
#include "map"
#include <cfloat>
#include "eigen3/Eigen/Dense"
#include "fstream"
#include "../tool/format.h"


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
vector<string> split(const string &str, const string &delim);
vector<wmMat> matReader(const string &path) {
    io::CSVReader<9> in(path);
    in.read_header(io::ignore_extra_column, "x", "y", "z", "c", "v", "b", "ux", "I", "A");
    double x, y, z, c, v, b, ux, I, A;
    vector<wmMat> mat;
    while (in.read_row(x, y, z, c, v, b, ux, I, A)) {
        wmMat row = {{x, y, z, c, v, b}, ux, (int) I, (int) A};
        mat.push_back(row);
    }
    return mat;
}

vector<vector<double>> rulesReader(const string &path) {
    io::CSVReader<2> in(path);
    in.read_header(io::ignore_extra_column, "x", "y");
    double x, y;
    vector<double> xs, ys;
    while (in.read_row(x, y)) {
        xs.push_back(x);
        ys.push_back(y);
    }
    return vector<vector<double>>{xs, ys};
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
    Eigen::ArrayXXf rules(samples, attribute - 1); // I was not needed
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
    vector<map<int, boundary>> boundaries(6); // x ,y ,.... 6 is the dimension of dataset
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
    vector<vector<double>> regions(6); // 6 is the number of attribute
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
        for (auto i=0;i<row.size();++i){
            if (i==row.size()-1){
                out << row[i];
            }else{
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


string strJoin(vector<string> &strs) {
    string res;
    for (const auto &str: strs) {
        res += str + " ";
    }
    return res;
}

string numJoin(vector<double> &nums){
    string res;
    for (const auto &num:nums){
        res += std::to_string(num)+" ";
    }
    return res;
}

vector<double> muSigma(double a,double b){
    double mu,sigma;
    if (b-a<100){
        mu = a+(b-a)/2;
        sigma = (b-a);
    }else{
        if (a<0){
            mu = b;
            sigma = b;
        }else{
            mu = a;
            sigma = a/8;
        }
    }
    if (sigma<=0){
        sigma = 1;
    }
    return vector<double>{sigma,mu};
}

void fuzzyMatlab(vector<vector<double>> &partition, map<string, map<int, float>> &model,vector<wmMat> &mat) {
    string fuzzyLogicCode = "fis = mamfis('Name',\"HDBSCANFS\");\n";
    int attributeName = 1000;
    for (const auto &attribute: partition) {
        string input = SFormat("fis = addInput(fis,[{0} {1}],'Name',\"{2}\");\n", 0, attribute[attribute.size() - 2],
                               attributeName);
        fuzzyLogicCode += input; // add attribute to code;
        char regionName = 67; // start from a
        for (int i = 0; i < attribute.size() - 1; ++i) {
            auto muSigmas = muSigma(attribute[i],attribute[i+1]);
            fuzzyLogicCode += SFormat("fis = addMF(fis,\"{0}\",\"{1}\",[{2} {3}],'Name',\"{4}\");\n", attributeName,
                                      "gaussmf", muSigmas[0], muSigmas[1], regionName); // addMF
            regionName++; // update
        }
        attributeName++; // update
    }
    fuzzyLogicCode += "fis = addOutput(fis,[0 83],'Name',\"mps\");\n";
    for (const auto &sample:mat){ // add output
        fuzzyLogicCode += SFormat("fis = addMF(fis,\"mps\",\"gaussmf\",[1,{0}],'Name',\"{1}\");\n",sample.A,attributeName);
        attributeName++;
    }
    fuzzyLogicCode += "ruleList = [";
    for (auto &rule: model) {
        string streamString;
        auto condition = split(rule.first, "-"); // if A and B
        auto result = rule.second.begin()->first; // then C
        condition.emplace_back(std::to_string(result)); // combine A B C
        condition.emplace_back("1"); // weight is constantly equal to 1 (one win)
        condition.emplace_back("1"); // AND
        fuzzyLogicCode += strJoin(condition)+";";
    }
    fuzzyLogicCode.pop_back(); //  remove last ;
    fuzzyLogicCode += "];\ninputs = [";
    for (auto sample:mat) {
        fuzzyLogicCode += numJoin(sample.x)+";";
    }
    fuzzyLogicCode += "];\nfis = addRule(fis,ruleList);\nevalfis(fis,inputs)";

    std::ofstream file;
    file.open("fuzzyLogic.m");
    file << fuzzyLogicCode;
    file.close();
//    cout << fuzzyLogicCode;
}

void modelClean(map<string, map<int, float>> &model) {
    vector<string> useless;
    for (const auto &rule: model) {
        for (const auto &key: rule.second) {
            if (key.first < 0 || key.first > 100) {
                useless.emplace_back(rule.first);
            }
        }
    }
    for (const auto &key: useless) {
        model.erase(key);
    }
}

vector<string> split(const string &str, const string &delim) {
    vector<string> res;
    if (str.empty()) return res;
    char *strs = new char[str.length() + 1];
    strcpy(strs, str.c_str());

    char *d = new char[delim.length() + 1];
    strcpy(d, delim.c_str());

    char *p = strtok(strs, d);
    while (p) {
        string s = p;
        res.push_back(s);
        p = strtok(nullptr, d);
    }

    return res;
}

int main() {
    // auto partition = rulesReader("../data/rules.csv"); // option for origin wm algorithm
    auto mat = matReader("../data/carwm.csv");
    auto partition = fuzzyRegion(mat);
    auto model = wang_mendel(partition, mat);
    auto test = matReader("../data/carwm.csv");
    auto result = predict(model, test, partition);
    to_csv("results.csv", result);
}