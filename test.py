import cv2
import dlib
import numpy as np
import os

# 初始化探测器和预测器（请确保 .dat 文件和这个脚本在同一个文件夹）
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def visualize_emotion(image_path):
    if not os.path.exists(image_path):
        print(f"找不到图片，请检查路径: {image_path}")
        return

    frame = cv2.imread(image_path)

    # 很多测试集图片太小(48x48)，为了画图和看字清晰，我们把它放大3倍
    h, w = frame.shape[:2]
    if w < 150:
        frame = cv2.resize(frame, (w * 3, h * 3))

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    h, w = gray_frame.shape

    # 检测人脸
    rects = detector(gray_frame, 1)
    if len(rects) > 0:
        rect = rects[0]
    else:
        # 兜底机制
        rect = dlib.rectangle(2, 2, w - 2, h - 2)

    # 提取68个关键点
    landmarks = np.array([[p.x, p.y] for p in predictor(gray_frame, rect).parts()])

    # === 计算特征 (保持与你主代码完全一致) ===
    d1 = np.linalg.norm(landmarks[37] - landmarks[41])
    d2 = np.linalg.norm(landmarks[38] - landmarks[40])
    d3 = np.linalg.norm(landmarks[36] - landmarks[39])
    c1 = (d1 + d2) / (2 * d3)

    d4 = np.linalg.norm(landmarks[65] - landmarks[60])
    d5 = np.linalg.norm(landmarks[67] - landmarks[64])
    d6 = np.linalg.norm(landmarks[48] - landmarks[54])
    c2 = (d4 + d5) / (2 * d6) if d6 > 0 else 0

    d7 = np.linalg.norm(landmarks[31] - landmarks[48])
    d8 = np.linalg.norm(landmarks[35] - landmarks[54])
    c3 = (d7 + d8) / (2 * d6) if d6 > 0 else 0

    d_eyes_outer = np.linalg.norm(landmarks[36] - landmarks[45])
    mouth_width_ratio = d6 / d_eyes_outer if d_eyes_outer > 0 else 0

    # === 判定逻辑 ===
    prediction = "UNKNOWN"
    color = (0, 255, 255)  # 默认黄色

    if c1 >= 0.12:
        if c3 >= 0.53:
            prediction = "SAD"
            color = (0, 0, 255)  # 红色表示 Sad
        elif (c2 >= 0.35 or mouth_width_ratio >= 0.72) and c3 < 0.51:
            prediction = "HAPPY"
            color = (0, 255, 0)  # 绿色表示 Happy
        elif c3 >= 0.51:
            prediction = "SAD"
            color = (0, 0, 255)

    # 1. 把 68 个特征点画在人脸上
    for (x, y) in landmarks:
        cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)  # 蓝色小圆点标出特征

    # 2. 准备要在图片上打印的文字，保留两位小数
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_c1 = f"EAR(c1): {c1:.2f}"
    text_c2 = f"MAR(c2): {c2:.2f}"
    text_c3 = f"NMR(c3): {c3:.2f}"
    text_pred = f"Pred: {prediction}"

    # 3. 将文字打在图片左上角
    cv2.putText(frame, text_c1, (5, 20), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, text_c2, (5, 40), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, text_c3, (5, 60), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    # 预测结果字号放大加粗
    cv2.putText(frame, text_pred, (5, 90), font, 0.7, color, 2, cv2.LINE_AA)

    # 保存图片到本地
    output_name = f"Report_Demo_{prediction}.jpg"
    cv2.imwrite(output_name, frame)
    print(f"成功！已生成演示图并保存为: {output_name}")

    # 弹出窗口展示给你看 (按任意键关闭)
    cv2.imshow("Visual Demonstration", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==========================================
# 使用指南：请把下面两条路径，替换成你测试集里的图片绝对路径
# 挑选一张标准的“大笑”，和一张标准的“悲伤”
# ==========================================

# 测试第一张：Happy
visualize_emotion(r"C:\Users\Admin\Desktop\Facial recognition\archive\test\happy\happy (1).jpg")

# 测试第二张：Sad
visualize_emotion(r"C:\Users\Admin\Desktop\Facial recognition\archive\test\sad\sad (4).jpg")