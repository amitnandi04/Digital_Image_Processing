import face_recognition as fr


known_image = fr.load_image_file("sc1.jpg")

unknown_img = fr.load_image_file("sc2.jpg")

known2_img = fr.load_image_file("lh1.jpg")

known_img_encoding = fr.face_encodings(known_image)[0]

unknown_img_encoding = fr.face_encodings(unknown_img)[0]

known2_img_encoding = fr.face_encodings(known2_img)[0]

result0 = fr.compare_faces([known_img_encoding], unknown_img_encoding)
result1 = fr.compare_faces([known_img_encoding], known2_img_encoding)
print()
print(result0)

print(result1)