import cv2
import numpy as np
import random
import time
import pygame

pygame.mixer.init()
pygame.mixer.music.load('/Users/shauryabhardwaj/Desktop/Stuff/Youtube/Music/Ryan_2.mp3')
pygame.mixer.music.play(-1, 0.0)

# Game area and configuration (full screen)
screen_width, screen_height = 1920, 1080
width, height = screen_width, screen_height
player_radius = 40
object_size = 60
motion_threshold = 2
speed_multiplier = 40
game_duration = 15
WIN_SCORE = 3
top_score = 0

collect_sound = pygame.mixer.Sound('/Users/shauryabhardwaj/Downloads/coin_sound.mp3')
game_over_sound = pygame.mixer.Sound('/Users/shauryabhardwaj/Downloads/dead_sound.mp3')
COIN_COOLDOWN = 0.3
flow_scale = 0.5

def draw_gradient_background(frame, start_color, end_color):
    for i in range(frame.shape[0]):
        alpha = i / frame.shape[0]
        color = (int(start_color[0]*(1 - alpha) + end_color[0]*alpha),
                 int(start_color[1]*(1 - alpha) + end_color[1]*alpha),
                 int(start_color[2]*(1 - alpha) + end_color[2]*alpha))
        cv2.line(frame, (0, i), (width, i), color, 1)

def draw_particles(frame, score):
    num_particles = 30 + score * 5
    for _ in range(num_particles):
        x = random.randint(0, width)
        y = random.randint(0, height)
        radius = random.randint(1, 4)
        color = (random.randint(0, 50), random.randint(0, 50), random.randint(200, 255))
        cv2.circle(frame, (x, y), radius, color, -1)

def draw_text_with_outline(frame, text, org, font_scale, text_color, outline_color, thickness=2):
    font = cv2.FONT_HERSHEY_COMPLEX
    for dx in [-2, -1, 0, 1, 2]:
        for dy in [-2, -1, 0, 1, 2]:
            if dx == 0 and dy == 0:
                continue
            cv2.putText(frame, text, (org[0] + dx, org[1] + dy), font, font_scale, outline_color, thickness)
    cv2.putText(frame, text, org, font, font_scale, text_color, thickness)

def flash_countdown():
    for i in range(3, 0, -1):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        text = str(i)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 4, 3)
        org = ((width - tw) // 2, (height + th) // 2)
        draw_text_with_outline(frame, text, org, 4, (0, 255, 255), (0, 0, 0), 3)
        cv2.imshow("Interactive Game", frame)
        cv2.waitKey(1000)

def post_menu(final_score, win):
    global top_score
    top_score = max(top_score, final_score)
    while True:
        menu_frame = np.zeros((height, width, 3), dtype=np.uint8)
        draw_gradient_background(menu_frame, (20, 20, 60), (80, 0, 80))
        message = "You Win!" if win else "Game Over!"
        main_color = (0, 255, 0) if win else (0, 0, 255)
        sub_message = f"Score: {final_score}"
        top_message = f"Top Score: {top_score}"
        (mw, mh), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_COMPLEX, 3, 3)
        (sw, sh), _ = cv2.getTextSize(sub_message, cv2.FONT_HERSHEY_COMPLEX, 2, 2)
        (tw, th), _ = cv2.getTextSize(top_message, cv2.FONT_HERSHEY_COMPLEX, 2, 2)
        msg_org = ((width - mw) // 2, height // 2 - 100)
        sub_org = ((width - sw) // 2, height // 2)
        top_org = ((width - tw) // 2, height // 2 + 60)
        draw_text_with_outline(menu_frame, message, msg_org, 3, main_color, (0, 0, 0), 3)
        draw_text_with_outline(menu_frame, sub_message, sub_org, 2, (255, 255, 255), (0, 0, 0), 2)
        draw_text_with_outline(menu_frame, top_message, top_org, 2, (255, 255, 0), (0, 0, 0), 2)
        cv2.putText(menu_frame, "Press 'R' to Restart", (width//2 - 250, height//2 + 150),
                    cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        cv2.putText(menu_frame, "Press 'Q' to Quit", (width//2 - 200, height//2 + 220),
                    cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 2)
        cv2.imshow("Interactive Game - Menu", menu_frame)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('r'):
            return True
        elif key == ord('q'):
            return False

def generate_coin(player_position):
    while True:
        x = random.randint(0, width - object_size)
        y = random.randint(0, height - object_size)
        if abs(x - player_position[0]) > 50 and abs(y - player_position[1]) > 50:
            return [x, y]

def play_game():
    player_position = [width // 2, height // 2]
    red_squares = [generate_coin(player_position) for _ in range(5)]
    score = 0
    last_collect_time = 0

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return score, False

    prev_frame = cv2.resize(prev_frame, (width, height))
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    flash_countdown()
    start_time = time.time()
    game_active = True

    while game_active:
        ret, curr_frame = cap.read()
        if not ret:
            break
        curr_frame = cv2.resize(curr_frame, (width, height))
        curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        small_prev = cv2.resize(prev_frame_gray, (int(width * flow_scale), int(height * flow_scale)))
        small_curr = cv2.resize(curr_frame_gray, (int(width * flow_scale), int(height * flow_scale)))
        flow = cv2.calcOpticalFlowFarneback(small_prev, small_curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mean_motion_x = np.mean(flow[..., 0]) * speed_multiplier / flow_scale
        mean_motion_y = np.mean(flow[..., 1]) * speed_multiplier / flow_scale

        if abs(mean_motion_x) > motion_threshold:
            player_position[0] += int(mean_motion_x)
        if abs(mean_motion_y) > motion_threshold:
            player_position[1] += int(mean_motion_y)

        player_position[0] = max(player_radius, min(width - player_radius, player_position[0]))
        player_position[1] = max(player_radius, min(height - player_radius, player_position[1]))

        current_time = time.time()
        if current_time - last_collect_time > COIN_COOLDOWN:
            for square in red_squares[:]:
                if (square[0] < player_position[0] < square[0] + object_size and
                    square[1] < player_position[1] < square[1] + object_size):
                    red_squares.remove(square)
                    score += 1
                    red_squares.append(generate_coin(player_position))
                    collect_sound.play()
                    last_collect_time = current_time

        elapsed_time = current_time - start_time
        remaining_time = max(0, int(game_duration - elapsed_time))

        game_frame = curr_frame.copy()
        overlay = game_frame.copy()
        draw_gradient_background(overlay, (10, 10, 50), (50, 0, 70))
        cv2.addWeighted(overlay, 0.4, game_frame, 0.6, 0, game_frame)
        draw_particles(game_frame, score)
        cv2.circle(game_frame, tuple(player_position), player_radius, (0, 255, 0), -1)
        for square in red_squares:
            cv2.rectangle(game_frame, (square[0], square[1]),
                          (square[0] + object_size, square[1] + object_size), (0, 0, 255), -1)
        draw_text_with_outline(game_frame, f"Score: {score}", (20, 60), 2, (255, 255, 0), (0, 0, 0), 3)
        draw_text_with_outline(game_frame, f"Time: {remaining_time}s", (width - 320, 60), 2, (255, 255, 0), (0, 0, 0), 3)

        bar_width = int((game_duration - elapsed_time) / game_duration * width)
        cv2.rectangle(game_frame, (0, height - 40), (bar_width, height), (0, 255, 0), -1)
        cv2.rectangle(game_frame, (0, height - 40), (width, height), (255, 255, 255), 3)

        cv2.imshow("Interactive Game", game_frame)
        prev_frame_gray = curr_frame_gray.copy()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            game_active = False
            break
        if elapsed_time >= game_duration:
            game_active = False

    cap.release()
    cv2.destroyAllWindows()
    win = score >= WIN_SCORE
    if not win:
        game_over_sound.play()
    return score, win

cv2.namedWindow("Interactive Game", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Interactive Game", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    score, win = play_game()
    if not post_menu(score, win):
        break

pygame.mixer.music.stop()
cv2.destroyAllWindows()
