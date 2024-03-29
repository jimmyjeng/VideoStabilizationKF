#ifndef DBSCAN_H
#define DBSCAN_H

#include <vector>

struct ClusterPoint{
    double x;
    double y;
};

int dbscan(const std::vector<ClusterPoint> &input, std::vector<int> &labels, double eps, int min);

#endif /*DBSCAN_H*/
