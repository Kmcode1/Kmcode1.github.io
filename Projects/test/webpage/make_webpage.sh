#cd /root/Netflix/myproject/data_preprocess/webpage/
#conda activate das

rm -r videos_jul
python compile_julien.py

rm -r baselines
python compile_baselines.py

rm -r videos_motion_transfer
python compile_motion_transfer.py

rm -r camera_First
python compile_camera_FirstFrame.py

rm -r camera_dynamic
python compile_camera_dynamic.py

rm -r videos_mesh
python compile_mesh.py

python make_webpage.py

#cd ../
#rm webpage.zip
#zip -r webpage.zip webpage
