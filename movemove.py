from evdev import UInput, ecodes as e

ui = UInput({
    e.EV_REL: [e.REL_X,e.REL_Y],
    e.EV_KEY: [e.BTN_LEFT,e.BTN_RIGHT]})

def move(dx,dy):
    ui.write(e.EV_REL, e.REL_X,int(dx))
    ui.write(e.EV_REL, e.REL_Y,int(dy))
def left_down():
    ui.write(e.EV_KEY, e.BTN_LEFT,1)
    ui.syn()

def left_up():
    ui.write(e.EV_KEY, e.BTN_LEFT,0)
    ui.syn()
def right_down():
    ui.write(e.EV_KEY, e.BTN_RIGHT,1)
    ui.syn()
def right_up():
    ui.write(e.EV_KEY, e.BTN_RIGHT,0)
    ui.syn()

