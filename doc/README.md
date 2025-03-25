# Instructions to build the continuationHYPAD website:

1. Install requirements from `requirements.txt`
	- ```pip install -r requirements.txt```

2. Clone two branches of the repository in two directiories, as follows:

	- ```folder/continuationHYPAD-main``` Contains the main branch.
	- ```folder/continuationHYPAD-gh-pages``` Contains the gh-pages branch (Important!).

3. Move into ```continuationHYPAD-main/doc``` directory.

4. Run the make command to run the sphinx documentation generator. This will update the folder ```continuationHYPAD-gh-pages/html``` with the compiled library.

``` bash
make html
```

5. Move into ```continuationHYPAD-gh-pages``` directory.

``` bash
cd ../../continuationHYPAD-gh-pages/
```

6. Remove the previous website information. Keep this file (README.md), the html folder and the Makefile file in the ```continuationHYPAD-gh-pages/``` directory

``` bash
rm -rf -v !(html|Makefile|README.md)
```


6. Move all the contents of the continuationHYPAD-gh-pages/html folder into the ```continuationHYPAD-gh-pages/``` folder  

``` bash
mv html/* .
```

7. add and commit the changes to the gh-pages branch, and push to the repository.

``` bash
git add . html/* .
```