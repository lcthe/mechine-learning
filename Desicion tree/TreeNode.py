import array

class TreeNode:
    size = 0
    def __init__(self, val, left, right):
        self.val = val
        self.left = left
        self.right = right

    def insert(self, node):
        if self == None:
            return TreeNode(1, None, None)
        parent = None
        current = self
        while current != None :
            if node.val < current.val:
                parent = current
                current = current.left
            elif node.val > current.val:
                parent = current
                current = current.right

        if node.val < parent.val :
            parent.left = node
        else:
            parent.right = node
        self.size+=1
        return

    def preOrder(self):
        if self == None:
            return
        print(self.val)
        if self.left != None: self.left.preOrder()
        if self.right != None: self.right.preOrder()

if __name__ == '__main__':
    root = TreeNode(1, None, None)
    node1 = TreeNode(2, None, None)
    node2 = TreeNode(1.5, None, None)
    ndoe4 = TreeNode(3, None, None)

    root.insert(node1)
    root.insert(node2)
    root.insert(ndoe4)

    root.preOrder()



