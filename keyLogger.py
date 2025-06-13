import logging

from pynput.keyboard import Key, Listener
import time
'''
 Mike Manno
 19 March 2024
 Last update: 6 April 2025
 Clarkson University / AFRL
 This a Keystroke logger that only monitors and records digraphs and the 6 features planned for ML
 KNOWN ISSUES: 
 Does not account for 2 keys down at the same time.
 
'''

default_digraphs = ["ac", "al", "an", "ar", "as", "at", "ea", "ed", "en", "er", "es", "ha", "he", "hi", "in", "io",
                    "is", "it", "le", "nd", "ng", "nt", "on", "or", "ou", "re", "se", "te", "th", "ti", "to"]

# 21-30 in the list
#more_digraphs = ["nd", "ng", "al", "ou", "le", "mi", "nt", "as", "ar", "io"]

# Also adding in digraphs that are more web specific, such as popular websites and urls.
# Specific user digraphs such as www, name (Mike), or other common words unique to a user
# mike, www, com, google, gmail, facebook, youtube, netflix, disneyplus
custom_digraphs = ["mi", "ww", "om", "oo", "ma", "ac", "ou", 'et', "pl"]

digraphs = default_digraphs + custom_digraphs

data_typed = []
log_file = "/Users/mannom/Desktop/keylog.csv"
start_time = None

logging.basicConfig(filename=log_file, level=logging.DEBUG, format=" %(message)s")

print("A/B  Dwell A              Dwell B              UD(AB)              DU(AB)              UU(AB)              DD(AB)")
logging.info("A/B,  Dwell A,              Dwell B,              UD(AB),              DU(AB),              UU(AB),              DD(AB)")

def on_press(key):
    global start_time
    if start_time is None:
        start_time = time.time()


def on_release(key):
    global start_time
    key_pressed = None 
    duration = None 
    key_down = None

    if start_time is not None:
        key_down = start_time
        key_up = time.time()
        duration = key_up - key_down
        #Ignore all non-characters keys.
        if hasattr(key, 'char') and key.char and key.char.isprintable():
            key_pressed = [str(key), str(duration), str(key_down), str(key_up)]  # Set key_pressed data
            start_time = None  # Reset the start time for the next press

    # Special case for space key
    if key == Key.space:
        if key_down is None: 
            key_down = time.time() 
        if duration is None:  
            key_up = time.time()  
            duration = key_up - key_down  
        if key_pressed is None:  
            key_pressed = [" ", str(duration), str(key_down), str(key_up)]


    if key_pressed is not None:
        data_typed.append(key_pressed)

    if str(len(data_typed)) == '2':

        digraph = str(data_typed[0][0]).replace("'", "") + str(data_typed[1][0]).replace("'", "")
        # Planned Features
        # Dwell A, Dwell B, UD(AB), DU(AB), UU(AB), DD(AB)
        dwell_a = str(data_typed[0][1])
        dwell_b = str(data_typed[1][1])
        # (Up/Down) for the time between the first key release (up) and the second key press (down),
        ud = str((float(data_typed[1][2]) - float(data_typed[0][3])));
        #(Down/Up) measures the time between the first key press(down) and the second key release(up)
        du = str((float(data_typed[1][3]) - float(data_typed[0][2])));
        #(Up/Up) captures the time between the first key release(up) and the second key release(up)
        uu = str((float(data_typed[1][2]) - float(data_typed[0][2])));
        #(Down/Down) indicates the time between the first key press(down) and the second key press(down)
        dd = str((float(data_typed[1][3]) - float(data_typed[0][3])));

        #First Key
        uda = str(data_typed[0][2])
        dua = str(data_typed[1][2])
        #Second Key
        udb = str(data_typed[0][3])
        dub = str(data_typed[1][3])

        # TODO We need to account for special characters such as spacebar, if pressed, the 2 letters are not consecutive and are not a digraph

        if any(x in digraph for x in digraphs):
            print(" " + digraph + ", " + dwell_a, ", " + dwell_b, ", " + ud + ", " + du + ", " + uu + ", " + dd)
            logging.info(digraph + ", " + str(dwell_a) + ", " + str(dwell_b) + ", " + str(ud) + ", " + str(du)
                         + ", " + str(uu) + ", " + str(dd))

        # Keep me at 2 elements
        data_typed.pop(0)

    # Stop listener when 'esc' is pressed
    if key == Key.esc:
        return False


with Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()


