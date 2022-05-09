# imports
import cv2

## Capturing the input
cap = cv2.VideoCapture('cameraVid.mp4')

## Setting up the video frame
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
out = cv2.VideoWriter("my_out.avi", fourcc, 5.0, (720, 424))

## Reading the video capture
ret, frame1 = cap.read()
ret, frame2 = cap.read()
print(frame1.shape)

## While the video feed is open do :
while cap.isOpened():

    # Read the frame 0
    (grabbed, frame0) = cap.read()

    # Gray_Frame : convert colors to scale 1 byte per pixel, making the computation easier
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Blur_Frame : Edge changes don't get affected w/ or w/o  blur
    blur1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    blur2 = cv2.GaussianBlur(gray2, (5, 5), 0)

    # Delta_Frame :  For every pixel we need to compare every pixel with the gray and getting the absolute difference.
    # It shows only the parts that is moving.If the value changes it appears white.
    diff = cv2.absdiff(blur1, blur2)

    # Threashold_Frame :it takes the gray_blurred_delta frame's pixels if the value of the color pixel is > 20 we give 255(white)
    #                       if <20 we give 0(Black)
    _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

    # Dilating_Frame : Expand the white pixels a little bit
    dilated = cv2.dilate(thresh, None, iterations=3)

    # Contour_Frame : find the contour of the moving object
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        #
        (x, y, w, h) = cv2.boundingRect(contour)

        #
        if cv2.contourArea(contour) < 1000:
            continue

        # Draw the contour on the moving object with the red color
        cv2.drawContours(frame1, [contour], 0, (0, 0, 255), 2)

        # Drawing a box on the moving object
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Status message if there's a moving object or not
        cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)

    # cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

    ## Resize the frames and displaying them in their window_frame
    image = cv2.resize(frame1, (720, 424))
    image1 = cv2.resize(frame0, (400, 236))
    imagex = cv2.resize(gray1, (400, 236))
    image2 = cv2.resize(blur1, (400, 236))
    image3 = cv2.resize(thresh, (400, 236))
    image4 = cv2.resize(dilated, (400, 236))
    image5 = cv2.resize(diff, (400, 236))
    out.write(image)
    cv2.imshow("feed", image)
    cv2.imshow("rawFeed", image1)
    cv2.imshow("grey", imagex)
    cv2.imshow('blur', image2)
    cv2.imshow('thresh', image3)
    cv2.imshow('dilated', image4)
    cv2.imshow('difference', image5)

    frame1 = frame2
    ret, frame2 = cap.read()

    ## Hotkey to shutdown the application
    if cv2.waitKey(40) == 27:
        break

## Destroying the window_frames and releasing the video feed
cv2.destroyAllWindows()
cap.release()

out.release()