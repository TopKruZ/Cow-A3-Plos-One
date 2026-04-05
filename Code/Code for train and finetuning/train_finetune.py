from ultralytics import YOLO

def main():
    model = YOLO(r"D:\Project\YoloV8\cow A3\Output\Model L 500\weights\best.pt")

    model.train(
        data=r"D:\Project\YoloV8\cow A3\dataset\data.yaml",
        epochs=200, 
        imgsz=640,
        batch=-1,
        lr0=0.00001,              
        device=0,
        single_cls=False,  
        classes=[4],            #[0 = drink , 3 = chinresting  , 4 = mounting]
        # Save folder
        name= r"D:\Project\YoloV8\cow A3\Output\Finetune_Mounting",
        save_period = 100
    )

if __name__ == "__main__":
    main()