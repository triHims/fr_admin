# import the necessary packages
from misc import FACIAL_LANDMARKS_68_IDXS
from misc import FACIAL_LANDMARKS_5_IDXS
from misc import shape_to_np
from misc import rect_to_bb
import numpy as np
import cv2

class FaceAligner:
        def __init__(self, predictor, desiredLeftEye=(0.35, 0.35),
                desiredFaceWidth=250, desiredFaceHeight=None):
                # store the facial landmark predictor, desired output left
                # eye position, and desired output face width + height
                self.predictor = predictor
                self.desiredLeftEye = desiredLeftEye
                self.desiredFaceWidth = desiredFaceWidth
                self.desiredFaceHeight = desiredFaceHeight

                # if the desired face height is None, set it to be the
                # desired face width (normal behavior)
                if self.desiredFaceHeight is None:
                        self.desiredFaceHeight = self.desiredFaceWidth

        def align(self, image, gray, rect):
                # convert the landmark (x, y)-coordinates to a NumPy array
                shape = self.predictor(gray, rect)
                shape = shape_to_np(shape)
                # return image

                #simple hack ;)
                if (len(shape)==68):
                        # extract the left and right eye (x, y)-coordinates
                        (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
                        (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
                else:
                        (lStart, lEnd) = FACIAL_LANDMARKS_5_IDXS["left_eye"]
                        (rStart, rEnd) = FACIAL_LANDMARKS_5_IDXS["right_eye"]

                leftEyePts = shape[lStart:lEnd]
                rightEyePts = shape[rStart:rEnd]

                # compute the center of mass for each eye
                leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
                rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

                # compute the angle between the eye centroids
                dY = rightEyeCenter[1] - leftEyeCenter[1]
                dX = rightEyeCenter[0] - leftEyeCenter[0]
                angle = np.degrees(np.arctan2(dY, dX)) - 180
        
                # compute the desired right eye x-coordinate based on the
                # desired x-coordinate of the left eye
                desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

                # determine the scale of the new resulting image by taking
                # the ratio of the distance between eyes in the *current*
                # image to the ratio of distance between eyes in the
                # *desired* image
                dist = np.sqrt((dX ** 2) + (dY ** 2))
                desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
                desiredDist *= self.desiredFaceWidth
                scale = desiredDist / dist

                # compute center (x, y)-coordinates (i.e., the median point)
                # between the two eyes in the input image
                eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                        (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

                # grab the rotation matrix for rotating and scaling the face
                M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

                # update the translation component of the matrix
                tX = self.desiredFaceWidth * 0.5
                tY = self.desiredFaceHeight * self.desiredLeftEye[1]
                M[0, 2] += (tX - eyesCenter[0])
                M[1, 2] += (tY - eyesCenter[1])

                # apply the affine transformation
                (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
                output = cv2.warpAffine(image, M, (w, h),flags=cv2.INTER_CUBIC)




                trans=np.array( [ (rect.left(),rect.top()), (rect.right(),rect.top()), (rect.right(),rect.bottom()), (rect.left(),rect.bottom()) ],np.int32)
                # print(M)
                nrect= cv2.transform(trans[None,:,:],M)
                
                
                xt=nrect[0][0][0]
                yt=nrect[0][0][1]
                xb=nrect[0][2][0]
                yb=nrect[0][2][1]

                squaredx=convert_to_square( np.array( [xt,yt,xb,yb] ) )
                output=output[squaredx[1]:squaredx[3], squaredx[0]:squaredx[2]]
                # print(squaredx)
                # cv2.imshow("img",output)
                # cv2.waitKey(0)
                # return the aligned face
                print("=================================================================")
                
                if(output.shape[0] < 1 or  output.shape[1] < 1 or output.shape[2] <1 ) :
                        print("(+_+) this frame was excluded due to invalid size ",output.shape[0],  output.shape[1] ,output.shape[2])
                        return None
                        
               
                return cv2.resize(output,(150,150))





def convert_to_square(bboxes):
    """Convert bounding boxes to a square form.
    Arguments:
        bboxes: a float numpy array of shape [n, 4].
    Returns:
        a float numpy array of shape [n, 4],
            squared bounding boxes.
    """

    
    square_bboxes = [0,0,0,0]
    x1, y1, x2, y2 = [bboxes[i] for i in range(4)]
    h = y2 - y1 + 1.0
    w = x2 - x1 + 1.0
    max_side = np.maximum(h, w)
    square_bboxes[0] = int(x1 + w * 0.5 - max_side * 0.5)
    square_bboxes[1] = int(y1 + h * 0.5 - max_side * 0.5)
    square_bboxes[2] = int(square_bboxes[0] + max_side - 1.0)
    square_bboxes[3] = int(square_bboxes[1] + max_side - 1.0)
    return square_bboxes