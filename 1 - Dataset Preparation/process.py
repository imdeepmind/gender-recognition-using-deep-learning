import cv2
import sys

def process_images(gender, iter):
    index = 1
    for i in range(1,iter):
        try:    
            path = 'rawData/' + gender + '/' + gender + '_' + str(i) + '.jpg'
            image = cv2.imread(path, 0)
            face_cascade = cv2.CascadeClassifier('../cascades/haarcascade_frontalface_alt2.xml')
            face = face_cascade.detectMultiScale(image, 1.1, 5)
            for x,y,w,h in face:
                roi = image[y:y+h,x:x+w]
                newPath = 'processedData/' + gender + '/' + gender + '_' + str(index) + '.jpg'
                resized_image = cv2.resize(roi, (64, 64)) 
                if cv2.imwrite(newPath, resized_image):
                    print('-- Sucessfully saved the ' + gender + ' image ' + str(i) + ' --')
                else:
                    print('-- Failed to saved the ' + gender + ' image ' + str(i) + ' --')
                index += 1
        except Exception as e:
            print('-- Something went wrong with the image number ' + str(i) + ' --')
            print(e)
            


def main():
    no_of_images = 1600
    process_images('girls', no_of_images)
    # process_images('boys', no_of_images)

if __name__ == "__main__":
    main()