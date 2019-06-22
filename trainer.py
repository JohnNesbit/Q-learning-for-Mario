import pyautogui
from keyboard import press, release
import time
import numpy as np
import tensorflow as tf
from PIL import Image as PIL
from collections import deque
import random

# variables declaring the saves made in the last training sess (if there was one)
already_ran = False
# the number after model in the path to the saved trained variables
train_time_num = 0
# the number after "Q-learning_model" in the saved variables file
batch_num = 0

sess = tf.InteractiveSession()
batch_size = 1
epochs = 2
GAMMA = .4
train_time = 11

output_to_hex = {
    0: "f", # jump
    1: "b", # right
    2: "c", # left
    3: "d", # shoot
    4: "v", #down
    5: ("f", "b"),
    6: ("d", "b"),
    7: ("f", "c"),
    8: ("d", "c")

}


# directx scan codes http://www.gamespp.com/directx/directInputKeyboardScanCodes.html

m0, m1, m2, m3, m4, m5, m6, m7, m8, m9 = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]),\
                                         np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
var_list = [m0, m1, m2, m3, m4, m5, m6, m7, m8, m9]

for i in range(10):
    exec('m%d = np.array(PIL.open("pics/m" + str(i) + ".png").convert("L"))' % i)


def ivar(shape, var_type):
    if var_type == "w":
        return tf.Variable(tf.truncated_normal(shape, stddev=0.01, dtype=tf.float64), dtype=tf.float64)
    if var_type == "b":
        return tf.Variable(tf.constant(0.01, shape=shape, dtype=tf.float64), dtype=tf.float64)


#946, 1025, 3
w1 = ivar([60, 65, 1, 1], "w")
#  473, 513, 3
b1 = ivar([1, 473, 513, 1], "b")
w2 = ivar([119, 129, 1, 1], "w")
b2 = ivar([1, 237, 257, 1], "b")
w3 = ivar([119, 129, 1, 1], "w")
b3 = ivar([1, 119, 129, 1], "b")
w4 = ivar([60, 65, 1, 1], "w")
b4 = ivar([1, 60, 65, 1], "b")
w5 = ivar([12, 13, 1, 1], "w")
b5 = ivar([1, 30, 33, 1], "b")
w6 = ivar([1, 1, 990, 330], "w")
b6 = ivar([1, 330], "b")
w7 = ivar([1, 1, 330, 110], "w")
b7 = ivar([1, 110], "b")
w8 = ivar([1, 1, 110, 11], "w")
b8 = ivar([1, 11], "b")
w9 = ivar([1, 1, 11, 8], "w")
b9 = ivar([8], "b")

current_state = pyautogui.screenshot().crop((447, 43, 1472, 989))


def do_action(action_int):
    if action_int == 1337:
        hexint = "enter"
        press(hexint)
        time.sleep(.5)
        release(hexint)

    else:
        if action_int >= 5:
            for q in output_to_hex[action_int]:
                press(q)
            time.sleep(.3)
            for rq in output_to_hex[action_int]:
                release(rq)
        else:
            hexint = output_to_hex[action_int]
            press(hexint)
            time.sleep(.5)
            release(hexint)


def get_reward(current_statel, past_reward):
    hundred_thousands_place = 0
    ten_thousands_place = 0
    thousands_place = 0
    hundreds_place = 0
    ten_thousands = np.array(current_statel.crop((96, 113, 125, 142)).convert("L"))
    ten_thousands[thousands_place < 254] = 0
    hundred_thousands = np.array(current_statel.crop((126, 113, 157, 142)).convert("L"))
    hundred_thousands[hundred_thousands < 254] = 0
    hundreds_arr = np.array(current_statel.crop((158, 113, 189, 142)).convert("L"))
    hundreds_arr[hundreds_arr < 254] = 0
    thousands_arr = np.array(current_statel.crop((192, 113, 221, 142)).convert("L"))
    thousands_arr[thousands_arr < 254] = 0

    # crops: 4-tuple defining the left, upper, right, and lower pixel coordinate.
    for num in range(10):
        x = eval('m%d' % num)
        x[x < 254] = 0

        if np.array_str(hundreds_arr) == np.array_str(x):

            hundreds_place = num

        if np.array_equal(x, thousands_arr):
            thousands_place = num

        if np.array_str(ten_thousands) == np.array_str(x):

            ten_thousands_place = num

        if np.array_equal(x, hundred_thousands):
            hundred_thousands_place = num

    score = int(str(ten_thousands_place) + str(hundred_thousands_place) + str(thousands_place) + str(hundreds_place))
    thousands_place = 0
    hundreds_place = 0
    hundreds_arr = np.array(current_statel.crop((608, 337, 637, 366)).convert("L")) #sub-level box
    thousands_arr = np.array(current_statel.crop((544, 337, 573, 366)).convert("L")) # level box
    # crops: 4-tuple defining the left, upper, right, and lower pixel coordinate.
    win_bonus = 0
    if current_statel.convert("L") == PIL.open("pics/win.png").convert("L"):
        win_bonus = 120
    for num in range(10):
        x = eval('m%d' % num)
        x[x < 254] = 0
        if np.array_str(hundreds_arr) == np.array_str(x):
            hundreds_place = num
        if np.array_equal(x, thousands_arr):
            thousands_place = num
    rew = int(str(thousands_place) + str(hundreds_place)) + win_bonus
    if rew == past_reward:
        return 0
    else:
        return rew + score


# everything up to this point has been tested and worksf

def is_game_finished(tstate):
    # returns bool if game over or post-won
    tprep = np.array_str(np.array(tstate.crop((0, 431, 1024, 945)).convert("L")))
    if tprep == \
            np.array_str(np.array(PIL.open("pics/game_over.png").crop((0, 431, 1024, 945)).convert("L"))):
        return True
    if tprep == \
            np.array_str(np.array(PIL.open("pics/post-win.png").crop((0, 431, 1024, 945)).convert("L"))):
        return True
    return False


def run_net(data):
    data = tf.reshape(data, shape=[1, 946, 1025, 1])
    f = tf.nn.conv2d(data, w1, strides=[1, 2, 2, 1], padding="SAME")
    f = tf.nn.relu(f + b1)
    f = tf.nn.conv2d(f, w2, strides=[1, 2, 2, 1], padding="SAME")
    f = tf.nn.relu(f + b2)
    f = tf.nn.conv2d(f, w3, strides=[1, 2, 2, 1], padding="SAME")
    f = tf.nn.relu(f + b3)
    f = tf.nn.conv2d(f, w4, strides=[1, 2, 2, 1], padding="SAME")
    f = tf.nn.relu(f + b4)
    f = tf.nn.conv2d(f, w5, strides=[1, 2, 2, 1], padding="SAME")
    f = tf.nn.relu(f + b5)
    f = tf.reshape(f, [1, 1, 1, 990])
    f = tf.matmul(f, w6)
    f = tf.nn.relu(f + b6)
    f = tf.reshape(f, [1, 1, 1, 330])
    f = tf.matmul(f, w7)
    f = tf.nn.relu(f + b7)
    f = tf.reshape(f, [1, 1, 1, 110])
    f = tf.matmul(f, w8)
    f = tf.nn.sigmoid(f + b8)
    f = tf.reshape(f, [1, 1, 1, 11])
    f = tf.matmul(f, w9)
    f = tf.reshape(f, [8])
    f = tf.nn.sigmoid(f + b9)
    return f # sigmoid of shape 5


x_placeholder = tf.placeholder(tf.float64, shape=[1, 946, 1025, 1], name='x_placeholder')
action_done = tf.placeholder(tf.float64, [None, 8], name="action_done")
y = tf.placeholder(tf.float64, [None])

TEST = run_net(x_placeholder)
readout_action = tf.reduce_sum(tf.multiply(TEST, action_done), reduction_indices=1)
cost = tf.reduce_mean(tf.square(y - readout_action))
train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)
epsilon = 1.0
sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()
if already_ran:
    saver.restore(sess, save_path="model" + train_time_num + "/Q-learning_model"+ batch_num +".ckpt")

for n in range(train_time):
    # observe phase
    for _ in range(batch_size):
        while True:
            current_state = np.array(pyautogui.screenshot().crop((447, 43, 1472, 989)).convert("L"))
            time.sleep(1)
            if np.array_str(current_state) == np.array_str(
                    np.array(pyautogui.screenshot().crop((447, 43, 1472, 989)).convert("L"))):
                press("enter")
                time.sleep(.5)
                release("enter")

            current_state = pyautogui.screenshot().crop((447, 43, 1472, 989))
            exec("D%d = deque()" % _)
            if random.random() < epsilon:
                action = random.randrange(0, 8, 1)
                epsilon -= .00001
            else:
                preaction = run_net(np.array(current_state.convert("L"), dtype=float))
                action = np.argmax(preaction.eval())
            print(action)
            do_action(action)
            new_state = pyautogui.screenshot().crop((447, 43, 1472, 989))
            if _ == 0:
                past_rew = 0
            else:
                past_rew = reward

            reward = get_reward(new_state, past_rew)
            print(reward)
            eval("D%d" % _).append((current_state.convert("L"), action, reward, np.array(new_state), is_game_finished(new_state)))

            if is_game_finished(new_state): # ends game if game over or if after win screen
                time.sleep(4)
                do_action(1337)
                time.sleep(4)
                break

    print('game' + str(n * batch_size) + 'completed')
    y_arr = np.array([])
    x_arr = np.array([])
    a_arr = np.array([])

    D = deque()

    # training by learning from observations
    for ___ in range(epochs):
        for iii in range(batch_size):
            D = eval("D%d" % iii)
            for xx in D:
                state = xx[0]

                action = xx[1]
                reward = xx[2]
                state_new = xx[3]
                done = xx[4]
                if not done:
                    y_arr = np.append(y_arr, (reward + GAMMA * np.argmax(run_net(state))))
                else:
                    y_arr = np.append(y_arr, reward)
                blank = np.zeros([1, 8])
                blank[0][action] = 1
                print(blank.shape)
                train_step.run(
                    feed_dict={
                    y: y_arr,
                    action_done: blank,
                    x_placeholder: np.reshape(state, [1, 946, 1025, 1])
                    }
                )

                y_arr = np.array([])

            if n % 5:
                saver.save(sess, 'model' + str(n) + '/Q-learning_model' + str(iii) + '.ckpt')

