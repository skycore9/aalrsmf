bsub -x -q normal -n 12 -a openmp -o netflix1.out -e netflix.err ./mf-train -l 0.05 -r 0.5 -e 0.01 -p ../dataset/netflix_mme.txt ../dataset/netflix_mm.txt
bsub -x -q normal -n 12 -a openmp -o netflix2.out -e netflix.err ./mf-train -l 0.05 -r 0.5 -e 0.001 -p ../dataset/netflix_mme.txt ../dataset/netflix_mm.txt
bsub -x -q normal -n 12 -a openmp -o netflix3.out -e netflix.err ./mf-train -l 0.05 -r 0.5 -e 0.0001 -p ../dataset/netflix_mme.txt ../dataset/netflix_mm.txt
bsub -x -q normal -n 12 -a openmp -o netflix4.out -e netflix.err ./mf-train -l 0.05 -r 0.5 -e 0.00001 -p ../dataset/netflix_mme.txt ../dataset/netflix_mm.txt
bsub -x -q normal -n 12 -a openmp -o netflix5.out -e netflix.err ./mf-train -l 0.05 -r 0.1 -e 0.01 -p ../dataset/netflix_mme.txt ../dataset/netflix_mm.txt
bsub -x -q normal -n 12 -a openmp -o netflix6.out -e netflix.err ./mf-train -l 0.05 -r 0.1 -e 0.001 -p ../dataset/netflix_mme.txt ../dataset/netflix_mm.txt
bsub -x -q normal -n 12 -a openmp -o netflix7.out -e netflix.err ./mf-train -l 0.05 -r 0.1 -e 0.0001 -p ../dataset/netflix_mme.txt ../dataset/netflix_mm.txt
bsub -x -q normal -n 12 -a openmp -o netflix8.out -e netflix.err ./mf-train -l 0.05 -r 0.1 -e 0.00001 -p ../dataset/netflix_mme.txt ../dataset/netflix_mm.txt
bsub -x -q normal -n 12 -a openmp -o netflix9.out -e netflix.err ./mf-train -l 0.05 -r 0.05 -e 0.01 -p ../dataset/netflix_mme.txt ../dataset/netflix_mm.txt
bsub -x -q normal -n 12 -a openmp -o netflix10.out -e netflix.err ./mf-train -l 0.05 -r 0.05 -e 0.001 -p ../dataset/netflix_mme.txt ../dataset/netflix_mm.txt
bsub -x -q normal -n 12 -a openmp -o netflix11.out -e netflix.err ./mf-train -l 0.05 -r 0.05 -e 0.0001 -p ../dataset/netflix_mme.txt ../dataset/netflix_mm.txt
bsub -x -q normal -n 12 -a openmp -o netflix12.out -e netflix.err ./mf-train -l 0.05 -r 0.05 -e 0.00001 -p ../dataset/netflix_mme.txt ../dataset/netflix_mm.txt

bsub -x -q normal -n 12 -a openmp -o movie1.out -e movie.err ./mf-train -l 0.05 -r 0.5 -e 0.01 -p ../dataset/movie_test.txt ../dataset/movie_train.txt
bsub -x -q normal -n 12 -a openmp -o movie2.out -e movie.err ./mf-train -l 0.05 -r 0.5 -e 0.001 -p ../dataset/movie_test.txt ../dataset/movie_train.txt
bsub -x -q normal -n 12 -a openmp -o movie3.out -e movie.err ./mf-train -l 0.05 -r 0.5 -e 0.0001 -p ../dataset/movie_test.txt ../dataset/movie_train.txt
bsub -x -q normal -n 12 -a openmp -o movie4.out -e movie.err ./mf-train -l 0.05 -r 0.5 -e 0.00001 -p ../dataset/movie_test.txt ../dataset/movie_train.txt
bsub -x -q normal -n 12 -a openmp -o movie5.out -e movie.err ./mf-train -l 0.05 -r 0.1 -e 0.01 -p ../dataset/movie_test.txt ../dataset/movie_train.txt
bsub -x -q normal -n 12 -a openmp -o movie6.out -e movie.err ./mf-train -l 0.05 -r 0.1 -e 0.001 -p ../dataset/movie_test.txt ../dataset/movie_train.txt
bsub -x -q normal -n 12 -a openmp -o movie7.out -e movie.err ./mf-train -l 0.05 -r 0.1 -e 0.0001 -p ../dataset/movie_test.txt ../dataset/movie_train.txt
bsub -x -q normal -n 12 -a openmp -o movie8.out -e movie.err ./mf-train -l 0.05 -r 0.1 -e 0.00001 -p ../dataset/movie_test.txt ../dataset/movie_train.txt
bsub -x -q normal -n 12 -a openmp -o movie9.out -e movie.err ./mf-train -l 0.05 -r 0.05 -e 0.01 -p ../dataset/movie_test.txt ../dataset/movie_train.txt
bsub -x -q normal -n 12 -a openmp -o movie10.out -e movie.err ./mf-train -l 0.05 -r 0.05 -e 0.001 -p ../dataset/movie_test.txt ../dataset/movie_train.txt
bsub -x -q normal -n 12 -a openmp -o movie11.out -e movie.err ./mf-train -l 0.05 -r 0.05 -e 0.0001 -p ../dataset/movie_test.txt ../dataset/movie_train.txt
bsub -x -q normal -n 12 -a openmp -o movie12.out -e movie.err ./mf-train -l 0.05 -r 0.05 -e 0.00001 -p ../dataset/movie_test.txt ../dataset/movie_train.txt

bsub -x -q normal -n 12 -a openmp -o music1.out -e music.err ./mf-train -l 1 -r 0.5 -e 0.01 -p ../dataset/music_test.txt ../dataset/music_train.txt
bsub -x -q normal -n 12 -a openmp -o music2.out -e music.err ./mf-train -l 1 -r 0.5 -e 0.001 -p ../dataset/music_test.txt ../dataset/music_train.txt
bsub -x -q normal -n 12 -a openmp -o music3.out -e music.err ./mf-train -l 1 -r 0.5 -e 0.0001 -p ../dataset/music_test.txt ../dataset/music_train.txt
bsub -x -q normal -n 12 -a openmp -o music4.out -e music.err ./mf-train -l 1 -r 0.5 -e 0.00001 -p ../dataset/music_test.txt ../dataset/music_train.txt
bsub -x -q normal -n 12 -a openmp -o music5.out -e music.err ./mf-train -l 1 -r 0.1 -e 0.01 -p ../dataset/music_test.txt ../dataset/music_train.txt
bsub -x -q normal -n 12 -a openmp -o music6.out -e music.err ./mf-train -l 1 -r 0.1 -e 0.001 -p ../dataset/music_test.txt ../dataset/music_train.txt
bsub -x -q normal -n 12 -a openmp -o music7.out -e music.err ./mf-train -l 1 -r 0.1 -e 0.0001 -p ../dataset/music_test.txt ../dataset/music_train.txt
bsub -x -q normal -n 12 -a openmp -o music8.out -e music.err ./mf-train -l 1 -r 0.1 -e 0.00001 -p ../dataset/music_test.txt ../dataset/music_train.txt
bsub -x -q normal -n 12 -a openmp -o music9.out -e music.err ./mf-train -l 1 -r 0.05 -e 0.01 -p ../dataset/music_test.txt ../dataset/music_train.txt
bsub -x -q normal -n 12 -a openmp -o music10.out -e music.err ./mf-train -l 1 -r 0.05 -e 0.001 -p ../dataset/music_test.txt ../dataset/music_train.txt
bsub -x -q normal -n 12 -a openmp -o music11.out -e music.err ./mf-train -l 1 -r 0.05 -e 0.0001 -p ../dataset/music_test.txt ../dataset/music_train.txt
bsub -x -q normal -n 12 -a openmp -o music12.out -e music.err ./mf-train -l 1 -r 0.05 -e 0.00001 -p ../dataset/music_test.txt ../dataset/music_train.txt
