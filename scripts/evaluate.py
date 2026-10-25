"""Select confidence on validation, evaluate held-out test, promote local weights."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import time

from pcb_inspector.constants import CLASS_NAMES
from pcb_inspector.data import parse_label
from pcb_inspector.training import match_detections, select_confidence, verify_manifest


def score(rows, threshold, class_id=None):
    tp=fp=fn=0
    for row in rows:
        predictions = [p for p in row['predictions'] if p['confidence'] >= threshold and (class_id is None or p['class_id']==class_id)]
        truths = [p for p in row['truths'] if class_id is None or p['class_id']==class_id]
        a,b,c = match_detections(predictions,truths)
        tp+=a; fp+=b; fn+=c
    precision=tp/(tp+fp) if tp+fp else 0.0
    recall=tp/(tp+fn) if tp+fn else 0.0
    return {'confidence':threshold,'tp':tp,'fp':fp,'fn':fn,'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0.0}


def predict_rows(model, prepared, split, cache_path, batch):
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    rows=[]
    stream=model.predict(source=str(prepared/split/'images'),stream=True,batch=batch,imgsz=640,conf=.05,iou=.50,device=0,verbose=False,max_det=300)
    for index,result in enumerate(stream):
        h,w=result.orig_shape
        stem=Path(result.path).stem
        truths=[]
        for c,x,y,bw,bh in parse_label((prepared/split/'labels'/(stem+'.txt')).read_text(encoding='utf-8-sig')):
            truths.append({'class_id':c,'bbox_xyxy':[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]})
        predictions=[]
        if result.boxes is not None:
            for x1,y1,x2,y2,confidence,c in result.boxes.data.cpu().tolist():
                predictions.append({'class_id':int(c),'confidence':confidence,'bbox_xyxy':[x1,y1,x2,y2]})
        rows.append({'filename':Path(result.path).name,'width':w,'height':h,'truths':truths,'predictions':predictions})
        if (index+1)%250==0:
            print(f'Predict {split}: {index+1} images',flush=True)
    cache_path.parent.mkdir(parents=True,exist_ok=True)
    cache_path.write_text(json.dumps(rows),encoding='utf-8')
    return rows


def ap_summary(result):
    classes={}
    for row,class_id in enumerate(result.box.ap_class_index.tolist()):
        classes[CLASS_NAMES[int(class_id)]]={'mAP50':float(result.box.all_ap[row,0]),'mAP50_95':float(result.box.all_ap[row].mean())}
    return {'mAP50':float(result.box.map50),'mAP50_95':float(result.box.map),'per_class':classes}


def save_examples(root, rows, threshold, directory):
    from PIL import Image,ImageDraw,ImageFont
    directory.mkdir(parents=True,exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    ranked=[]
    for row in rows:
        pred=[p for p in row['predictions'] if p['confidence']>=threshold]
        tp,fp,fn=match_detections(pred,row['truths'])
        ranked.append((fn+fp,fp,fn,tp,row,pred))
    panels=[]
    errors=sorted(ranked,key=lambda item:(item[0],item[2]),reverse=True)[:6]
    examples=sorted(ranked,key=lambda item:(item[3],-item[0]),reverse=True)[:6]
    for kind,selection in (('errors',errors),('detections',examples)):
        canvas=Image.new('RGB',(1260,1440),'#f4f7fc')
        for index,(_,fp,fn,tp,row,pred) in enumerate(selection):
            with Image.open(root/'data/prepared/test/images'/row['filename']) as source:
                image=source.convert('RGB')
            draw=ImageDraw.Draw(image)
            for truth in row['truths']:
                draw.rectangle(truth['bbox_xyxy'],outline='#c2410c',width=2)
            for detection in pred:
                box=detection['bbox_xyxy']
                draw.rectangle(box,outline='#1d4ed8',width=3)
                text=f'{CLASS_NAMES[detection["class_id"]]} {detection["confidence"]:.2f}'
                draw.text((box[0],max(0,box[1]-20)),text,font=font,fill='#1d4ed8',stroke_width=1,stroke_fill='white')
            image.thumbnail((400,400))
            x,y=(index%3)*420+10,(index//3)*720+58
            canvas.paste(image,(x,y))
            caption=ImageDraw.Draw(canvas)
            caption.text((x,y-43),f'TEST / TP {tp} FP {fp} FN {fn}',font=font,fill='#111827')
            caption.text((x,y+410),row['filename'][:39],font=font,fill='#111827')
            caption.text((x,y+438),'BLUE prediction / ORANGE ground truth',font=font,fill='#111827')
        canvas.save(directory/(f'test-{kind}.png'))
    return [{'filename':r[4]['filename'],'tp':r[3],'fp':r[1],'fn':r[2]} for r in errors]


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch',type=int,default=4)
    args=parser.parse_args()
    training=json.loads((root/'reports/training-config.json').read_text())
    checkpoint=Path(training['best_checkpoint'])
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != training['best_checkpoint_sha256']:
        raise RuntimeError('Checkpoint berubah sejak training; provenance tidak cocok')
    prepared=root/'data/prepared'
    manifest_path=prepared/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    verify_manifest(manifest)
    manifest_digest=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    from ultralytics import YOLO
    model=YOLO(str(checkpoint))
    if tuple(model.names[i] for i in range(6))!=CLASS_NAMES or len(model.names)!=6:
        raise RuntimeError('Mapping kelas checkpoint tidak sesuai')
    cache=root/'.local/evaluation-cache'/f'{digest[:12]}-{manifest_digest[:12]}'
    val_rows=predict_rows(model,prepared,'val',cache/'val.json',args.batch)
    curve=[score(val_rows,round(.10+i*.05,2)) for i in range(13)]
    selected=select_confidence(curve,split='val')
    threshold=selected['confidence']
    print('SELECTED_ON_VALIDATION',json.dumps(selected),flush=True)
    common=dict(data=str(prepared/'dataset.yaml'),batch=args.batch,imgsz=640,device=0,workers=0,conf=.001,iou=.50,plots=True,project=str(root/'runs'),exist_ok=False,verbose=False)
    validation_ap=ap_summary(model.val(split='val',name='validation-final',**common))
    # Test is consumed only after checkpoint and threshold have been frozen above.
    test_ap=ap_summary(model.val(split='test',name='test-final',**common))
    test_rows=predict_rows(model,prepared,'test',cache/'test.json',args.batch)
    fixed=score(test_rows,threshold)
    per_class={name:{**score(test_rows,threshold,i),**test_ap['per_class'].get(name,{})} for i,name in enumerate(CLASS_NAMES)}
    error_examples=save_examples(root,test_rows,threshold,root/'reports/visuals')
    evaluation={'checkpoint_sha256':digest,'manifest_sha256':manifest_digest,'class_names':list(CLASS_NAMES),'validation':{'selected_confidence':selected,'confidence_candidates':curve,**validation_ap},'test':{**fixed,**test_ap,'per_class':per_class,'images':len(test_rows)},'error_examples':error_examples,'metric_notes':'Fixed-threshold precision/recall/F1 use greedy class-aware IoU >= 0.5 matching. AP uses Ultralytics score curves (conf floor 0.001); no test-driven threshold tuning. Family-disjoint split differs from Kaggle original split.'}
    (root/'reports/evaluation.json').write_text(json.dumps(evaluation,indent=2),encoding='utf-8')
    models=root/'models'
    models.mkdir(exist_ok=True)
    shutil.copy2(checkpoint,models/'pcb-best.pt')
    metadata={'name':'YOLO11n PCB','checkpoint_sha256':digest,'class_names':list(CLASS_NAMES),'default_confidence':threshold,'pretrained_source':'Ultralytics yolo11n.pt / COCO','training_method':'full fine-tuning on PCB dataset','manifest_sha256':manifest_digest,'input_size':640,'nms_iou':.5,'training':training,'validation':evaluation['validation'],'test':evaluation['test'],'dataset_license':'Unknown on Kaggle; checkpoint remains local'}
    (models/'model-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('EVALUATION_COMPLETE',json.dumps(evaluation),flush=True)


if __name__=='__main__':
    main()
