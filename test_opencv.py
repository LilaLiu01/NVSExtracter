import cv2

# Load an image
img = cv2.imread('test.jpg')

if img is None:
    print("Image not found. Make sure 'test.jpg' is in your project folder.")
else:
    # Display image in a named window
    cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)  # makes it resizable
    cv2.imshow("Preview", img)

    print("Press any key inside the image window to close it.")
    key = cv2.waitKey(0)  # waits for a key press

    # Optional: if you want to close with Esc only
    # while True:
    #     if cv2.waitKey(1) & 0xFF == 27:  # 27 is ESC
    #         break

    cv2.destroyAllWindows()
