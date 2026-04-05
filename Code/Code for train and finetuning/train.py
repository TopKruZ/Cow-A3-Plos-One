from ultralytics import YOLO

def main():
    model = YOLO('D:\Project\YoloV8\cow A3\yolov8l.pt')

    model.train(
        data='D:\Project\YoloV8\cow A3\dataset\data.yaml',        
        epochs=500,        
        imgsz=640,            
        batch=16,               
        patience=0,             
        name='D:\Project\YoloV8\cow A3\Output\Model L 500 ',  
        lr0=0.001,               
        device="cuda",           
        save_period = 100
    )

if __name__ == '__main__':
    main()