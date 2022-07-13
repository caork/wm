#include <iostream>
#include "csv.h"
#include "map"
#include <cfloat>
#include "eigen3/Eigen/Dense"

using namespace std;

struct wmMat {
    vector<double> x;
    vector<double> ux;
    int I;
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
    io::CSVReader<5> in(path);
    in.read_header(io::ignore_extra_column, "x", "y", "ux", "uy", "I");
    double x, y, ux, uy, I;
    vector<wmMat> mat;
    while (in.read_row(x, y, ux, uy, I)) {
        wmMat row = {{x, y}, {ux, uy}, (int) I};
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

pairs fastSearch(double x, double y, vector<vector<double>> &partition) {
    int low = 0;
    int xRegion = binarySearch(x, partition[0], low, (int) partition[0].size() - 2);
    int yRegion = binarySearch(y, partition[1], low, (int) partition[0].size() - 2);
    return pairs{xRegion, yRegion};
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

void wang_mendel(vector<vector<double>> &partition, vector<wmMat> &mat) {
    int len = (int) partition[0].size();
    auto member = Eigen::ArrayXXf::Zero(len, len);
    vector<Eigen::ArrayXXf> membership = {member,member};
    for (auto xy: mat) {
        pairs result = fastSearch(xy.x[0], xy.x[1], partition);
        membership[0](result.x,result.y) += (float)xy.ux[0];
        membership[1](result.x,result.y) += (float)xy.ux[1];
    }
    for (int i = 0; i < len; ++i) {
        for (int j = 0; j < len; ++j) {
            if (membership[0](i,j) == 0.0 && membership[1](i,j) == 0.0) {
                continue;
            }
            char r = membership[0](i,j) > membership[1](i,j) ? 'x' : 'y';
            cout << "if " << i << " and " << j << " then " << r << endl;
            cout << "x: " << membership[0](i,j) << endl;
            cout << "y: " << membership[1](i,j) << endl;
            cout << "partitionX: " << partition[0][i] << " - " << partition[0][i + 1] << " partitionY "
                 << partition[1][j]
                 << " - " << partition[1][i + 1] << endl;
        }
    }
    for (auto &i: partition) {
        for (auto element: i) {
            cout << element << " ";
        }
        cout << endl;
    }
}

vector<vector<double>> fuzzyRegion(vector<wmMat> &mat) {
    vector<map<int, boundary>> boundaries(2); // x ,y ,.... 2 is the dimension of dataset
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
    vector<vector<double>> regions(2);
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


int main() {
    // auto partition = rulesReader("rules.csv"); // option for origin wm algorithm
    auto mat = matReader("result.csv");
    auto partition = fuzzyRegion(mat);
    wang_mendel(partition, mat);

}