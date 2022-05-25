# This is a sample Python script.
import array
from TreeNode import TreeNode



if __name__ == '__main__':
        root = TreeNode(1,None,None)
        node1 = TreeNode(2,None,None)
        node2 = TreeNode(1.5,None,None)
        ndoe4 = TreeNode(3,None,None)

        root.insert(node1)
        root.insert(node2)
        root.insert(ndoe4)

        root.preOrder()


