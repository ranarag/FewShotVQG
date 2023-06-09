import json


testing = [574783001, 581929001, 468457011, 487943000, 181873000, 374734000, 235522007, 308587012, 110434004, 530619000, 410612002, 422700035, 536110019, 82338005, 7072002, 442367004, 546928000, 38828002, 158127001, 530765003, 173208002, 398534014, 66445002, 553326000, 578524007, 330954002, 467297001, 186147002, 205514003, 171603000]

with open('data/vqa/v2_OpenEnded_mscoco_val2014_questions.json', 'r') as fid:
    asd = json.loads(fid.read())

    
with open('data/vqa/v2_mscoco_val2014_annotations.json', 'r') as fid:
    anno = json.loads(fid.read())
print(anno['annotations'][0])

# for question in asd['questions']:
# #         print(question.keys())
# #     if question['question_id'] in testing:
#         qid = question['question_id']
#         for tid in testing:
#             if qid == tid:
#                 print(question['question'])
# print(asd)
