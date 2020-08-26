import numpy as np
import time
import numpy as np
import copy
from operator import itemgetter


class BFS:  # 宽度优先
    def __init__(self, state, dirFlag=None, parent=None):
        self.state = state
        # state = np.array([[2, 8, 3], [1, Empty , 4], [7, 6, 5]])
        self.direction = ['up', 'down', 'right', 'left']
        if dirFlag:
            self.direction.remove(dirFlag)
        # record the possible directions to generate the sub-states
        self.parent = parent
        self.symbol = ' '

    def getDirection(self):
        return self.direction

    def showInfo(self):
        for i in range(3):
            for j in range(3):
                print(self.state[i, j], end='  ')
            print("\n")
        print('->')
        return

    def getEmptyPos(self):
        postion = np.where(self.state == self.symbol)
        return postion

    def generateSubStates(self):
        if not self.direction:
            return []
        subStates = []
        boarder = len(self.state) - 1
        # the maximum of the x,y
        row, col = self.getEmptyPos()  # 获取此时Empty所在行、列
        if 'left' in self.direction and col > 0:
            # it can move to left
            s = self.state.copy()
            temp = s.copy()
            s[row, col] = s[row, col-1]
            s[row, col-1] = temp[row, col]
            news = BFS(s, dirFlag='right', parent=self)
            subStates.append(news)
        if 'up' in self.direction and row > 0:
            # it can move to upper place
            s = self.state.copy()
            temp = s.copy()
            s[row, col] = s[row-1, col]
            s[row-1, col] = temp[row, col]
            news = BFS(s, dirFlag='down', parent=self)
            subStates.append(news)
        if 'down' in self.direction and row < boarder:        # it can move to down place
            s = self.state.copy()
            temp = s.copy()
            s[row, col] = s[row+1, col]
            s[row+1, col] = temp[row, col]
            news = BFS(s, dirFlag='up', parent=self)
            subStates.append(news)
        if self.direction.count('right') and col < boarder:    # it can move to right place
            s = self.state.copy()
            temp = s.copy()
            s[row, col] = s[row, col+1]
            s[row, col+1] = temp[row, col]
            news = BFS(s, dirFlag='left', parent=self)
            subStates.append(news)
        return subStates

    def solve(self):  # s1 = [[2, 8, 3], [1, Empty , 4], [7, 6, 5]]
        global originState
        # 定义open表
        openTable = []
        # 定义close表
        closeTable = []
        # append the origin state to the openTable
        openTable.append(self)
        steps = 1
        # start the loop
        while len(openTable) > 0:
            n = openTable.pop(0)
            closeTable.append(n)
            subStates = n.generateSubStates()
            path = []
            for s in subStates:
                if (s.state == s.answer).all():
                    while s.parent and s.parent != originState:
                        path.append(s.parent)
                        s = s.parent
                    path.reverse()
                    return path, steps+1
            openTable.extend(subStates)
            steps += 1
        else:
            return None, None


class DFS(object):
    directions = ['up', 'down', 'left', 'right']
    max = 7

    def __init__(self,arr,cost=0,parent=None):
        self.arr = arr
        self.cost = cost
        self.parent = parent

    def getCost(self):
        return self.cost

    def calc(self, state):  # 返回两个数组对应位置相同值的个数
        final = np.array([[1, 2, 3], [8, 0, 4], [7, 6, 5]])
        postion = np.where(state.arr == final)
        return len(state.arr[postion])

    # 打印八数码
    def showInfo(self):
        for i in range(3):
            for j in range(3):
                print(self.arr[i, j], end='   ')
            print("\n")
        print('->')

    def calc2(self, state1, stop):
        for x in stop:
            postion = np.where(state1.arr == x.arr)
            if len(state1.arr[postion]) == 9:
                return True
        return False

    def SubStates(self):
        subStates = []
        row, col = np.where(self.arr==0)
        for direction in self.directions:
            if 'left' == direction and col > 0:
                s = self.arr.copy()
                s[row, col],s[row, col - 1] = s[row, col - 1],s[row, col]
                new = DFS(s,self.cost+1,self)
                subStates.append(new)
            if 'up'  == direction and row > 0:
                s = self.arr.copy()
                s[row, col],s[row - 1, col] = s[row - 1, col],s[row, col]
                new = DFS(s, self.cost + 1,self)
                subStates.append(new)
            if 'down'  == direction and row < 2:
                s = self.arr.copy()
                s[row, col],s[row + 1, col] = s[row + 1, col],s[row, col]
                new = DFS(s, self.cost + 1,self)
                subStates.append(new)
            if 'right'  == direction and col < 2:
                s = self.arr.copy()
                s[row, col],s[row, col + 1] = s[row, col + 1],s[row, col]
                new = DFS(s, self.cost + 1,self)
                subStates.append(new)
        return subStates

    def DFS(self):
        stack = []
        stop = []
        stack.append(self)
        count = -1
        while True:
            if not stack:
                return False, count, node
            count += 1
            # stack = sorted(stack, key=self.calc)
            node = stack.pop()
            stop.append(node)
            # node.showInfo()
            if self.calc(node) == 9:
                return True,count,node
            s = node.SubStates()
            if s:
                res = sorted(s, key=self.calc)
            else:
                continue
            for x in res:
                if (x.cost + 9 - self.calc(x))< DFS.max:
                    if self.calc2(x,stop):
                        continue
                    stack.append(x)


def show_DFS(result):
    for node in result:
        for i in range(3):
            for j in range(3):
                print(node.arr[i, j], end='   ')
            print('\n')
        print('->')


# A*算法
def get_location(vec, num):  # 根据num元素获取num在矩阵中的位置
    row_num = vec.shape[0]  # numpy-shape函数获得矩阵的维数
    line_num = vec.shape[1]

    for i in range(row_num):
        for j in range(line_num):
            if num == vec[i][j]:
                return i, j


def get_actions(vec):  # 获取当前位置可以移动的下一个位置，返回移动列表
    row_num = vec.shape[0]
    line_num = vec.shape[1]

    (x, y) = get_location(vec, 0)  # 获取0元素的位置
    action = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    if x == 0:  # 如果0在边缘则依据位置情况，减少0的可移动位置
        action.remove((-1, 0))
    if y == 0:
        action.remove((0, -1))
    if x == row_num - 1:
        action.remove((1, 0))
    if y == line_num - 1:
        action.remove((0, 1))

    return list(action)


def result(vec, action):  # 移动元素，进行矩阵转化
    (x, y) = get_location(vec, 0)  # 获取0元素的位置
    (a, b) = action  # 获取可移动位置

    n = vec[x + a][y + b]  # 位置移动，交换元素
    s = copy.deepcopy(vec)
    s[x + a][y + b] = 0
    s[x][y] = n

    return s


def get_ManhattanDis(vec1, vec2):  # 计算两个矩阵的曼哈顿距离,vec1为目标矩阵,vec2为当前矩阵
    row_num = vec1.shape[0]
    line_num = vec1.shape[1]
    dis = 0

    for i in range(row_num):
        for j in range(line_num):
            if vec1[i][j] != vec2[i][j] and vec2[i][j] != 0:
                k, m = get_location(vec1, vec2[i][j])
                d = abs(i - k) + abs(j - m)
                dis += d

    return dis


def expand(p, actions, step):  # actions为当前矩阵的可扩展状态列表,p为当前矩阵,step为已走的步数
    children = []  # children用来保存当前状态的扩展节点
    for action in actions:
        child = {}
        child['parent'] = p
        child['vec'] = (result(p['vec'], action))
        child['dis'] = get_ManhattanDis(goal['vec'], child['vec'])
        child['step'] = step + 1  # 每扩展一次当前已走距离加1
        child['dis'] = child['dis'] + child['step']  # 更新该节点的f值  f=g+h（step+child[dis]）
        child['action'] = get_actions(child['vec'])
        children.append(child)

    return children


def node_sort(nodelist):  # 按照节点中字典的距离字段对列表进行排序,从大到小
    return sorted(nodelist, key=itemgetter('dis'), reverse=True)


def get_parent(node):
    q = {}
    q = node['parent']
    return q


def test():
    openlist = []  # open表
    close = []  # 存储扩展的父节点
    num = 3
    A = [[2, 8, 3], [1, 0, 4], [7, 6, 5]]
    B = [[1, 2, 3], [8, 0, 4], [7, 6, 5]]
    resultfile = '1.txt'
    goal['vec'] = np.array(B)  # 建立矩阵
    p = {}
    p['vec'] = np.array(A)
    p['dis'] = get_ManhattanDis(goal['vec'], p['vec'])
    p['step'] = 0
    p['action'] = get_actions(p['vec'])
    p['parent'] = {}

    if (p['vec'] == goal['vec']).all():
        return

    openlist.append(p)
    st = time.perf_counter()  # 开始扩展时CPU开始计算

    while openlist:
        children = []
        node = openlist.pop()  # node为字典类型，pop出open表的最后一个元素
        close.append(node)  # 将该元素放入close表

        if (node['vec'] == goal['vec']).all():  # 比较当前矩阵和目标矩阵是否相同
            end = (time.perf_counter() - st) * 1000  # CPU结束计算
            h = open(resultfile, 'w', encoding='utf-8', )  # 将结果写入文件  并在控制台输出
            h.write('搜索树规模：' + str(len(openlist) + len(close)) + '\n')
            h.write('close：' + str(len(close)) + '\n')
            h.write('openlist：' + str(len(openlist)) + '\n')
            h.write('cpu运行时间：' + str(end - st) + '微秒' + '\n')
            h.write('路径长：' + str(node['dis']) + '\n')
            h.write('解的路径：' + '\n')
            i = 0
            way = []
            while close:
                way.append(node['vec'])  # 从最终状态开始依次向上回溯将其父节点存入way列表中
                node = get_parent(node)
                if (node['vec'] == p['vec']).all():
                    way.append(node['vec'])
                    break
            while way:
                i += 1
                h.write(str(i) + '\n')
                h.write(str(way.pop()) + '\n')
            h.close()
            f = open(resultfile, 'r', encoding='utf-8', )
            print(f.read())

            return

        children = expand(node, node['action'], node['step'])  # 如果不是目标矩阵，对当前节点进行扩展，取矩阵的可能转移情况
        for child in children:  # 如果转移之后的节点，既不在close表也不再open表则插入open表，如果在close表中则舍弃，如果在open表则比较这两个矩阵的f值，留小的在open表
            f = False
            flag = False
            j = 0
            for i in range(len(openlist)):
                if (child['vec'] == openlist[i]['vec']).all():
                    j = i
                    flag = True
                    break
            for i in range(len(close)):
                if (child['vec'] == close[i]).all():
                    f = True
                    break
            if f == False and flag == False:
                openlist.append(child)
            elif flag == True:
                if child['dis'] < openlist[j]['dis']:
                    del openlist[j]
                    openlist.append(child)
        openlist = node_sort(openlist)  # 对open表进行从大到小排序


def main_BFS():
    print('宽度优先搜索')
    start = time.perf_counter()
    Empty = ' '

    BFS.symbol = Empty
    # 设置起始节点
    global originState
    originState = BFS(np.array([[2, 8, 3], [1, Empty, 4], [7, 6, 5]]))
    # 设置目标结点
    BFS.answer = np.array([[1, 2, 3], [8, BFS.symbol, 4], [7, 6, 5]])
    s1 = BFS(state=originState.state)  # s1 = [[2, 8, 3], [1, Empty , 4], [7, 6, 5]]
    path, steps = s1.solve()
    if path:  # if find the solution
        for node in path:
            # print the path from the origin to final state
            node.showInfo()
        print(BFS.answer)
        print("经过%d次变换结束" % steps)
    end = time.perf_counter() - start
    print('程序运行时间为:{:.2f}微秒'.format(end * 1000))


def main_DFS():
    print('-'*50)
    print('深度优先搜索')
    st = time.perf_counter()
    start = np.array([[2, 8, 3], [1, 0, 4], [7, 6, 5]])
    p = DFS(start)
    res, count, node = p.DFS()
    result = []
    if res:
        while node:
            result.append(node)
            node = node.parent
        result.reverse()
        show_DFS(result)
        print('经过%d次变换结束' % count)
    else:
        print('规定范围内未找到合适路径，可增大界值')
    end = time.perf_counter() - st
    print('程序运行时间为:{:.2f}微秒'.format(end * 1000))


def main_Astar():
    print('-'*50)
    print('A*搜索算法')
    test()


if __name__ == '__main__':
    goal = {}
    main_BFS()
    main_DFS()
    main_Astar()

