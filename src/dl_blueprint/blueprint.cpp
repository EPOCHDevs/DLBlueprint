//
// Created by dewe on 3/17/24.
//

#include "blueprint.h"


namespace dlb {

    void Blueprint::forward(TensorDict &tensors) {
        BFSVisit([&](NodeImpl *node) {
            AssertIfFalse(tensors.contains(node->key), "Fatal Error: Potential Multiple pass of the same module.");
            tensors.insert(node->key, node->module->forward(tensors[node->input]));
        });
    }

    void Blueprint::BFSVisit(Visitor const &visitor) {
        std::queue<NodeImpl *> queue;
        std::unordered_set<std::string> visited;

        visited.insert(m_root->key);
        queue.emplace(m_root.get());

        while (!queue.empty()) {
            auto current = queue.front();
            queue.pop();

            visitor(current);

            for (auto &child: current->children | std::views::values) {
                if (visited.count(child->key) == 0) {
                    visited.insert(child->key);
                    queue.emplace(child.get());
                }
            }
        }
    }
}