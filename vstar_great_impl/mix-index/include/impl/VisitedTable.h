/*
 * -------------------------------------------------------------------------
 * This file is part of the IndexSDK project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * IndexSDK is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 */


#ifndef IVFSP_VISITEDTABLE_H
#define IVFSP_VISITEDTABLE_H
#include <cstdint>
#include <iostream>
#include <vector>
#include <memory>
#include <cstring>

struct VisitedTable {
    std::vector<uint8_t> visited;
    uint8_t visno;

    explicit VisitedTable(int size = 100000) : visited(size), visno(1)
    {
    }

    // / set flag #no to true
    void set(int node)
    {
        visited[node] = visno;
    }

    // / get flag #no
    bool get(int node) const
    {
        return visited[node] == visno;
    }

    // / reset all flags to false
    void advance()
    {
        visno++;
        // 250 rather than 255 because sometimes we use visno and visno+1
        if (visno == 250) {
            auto ret = memset_s(visited.data(), sizeof(visited[0]) * visited.size(), 0,
                                sizeof(visited[0]) * visited.size());
            if (ret != 0) {
                std::cerr << "Error in VisitedTable:advance with memset; ret = " << ret << std::endl;
                return;
            }
            visno = 1;
        }
    }
};

#endif  // IVFSP_VISITEDTABLE_H
