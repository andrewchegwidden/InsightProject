# InsightProject

https://help.github.com/articles/adding-an-existing-project-to-github-using-the-command-line/
https://help.github.com/articles/adding-a-file-to-a-repository-using-the-command-line/

git init
# Only done once

git add
# Adds the files in the local repository and stages them for commit. To unstage a file, use 'git reset HEAD YOUR-FILE'.

git commit -m "message"
# Commits the tracked changes and prepares them to be pushed to a remote repository. To remove this commit and modify the file, use 'git reset --soft HEAD~1' and commit and add the file again.


git remote add origin https://github.com/andrewchegwidden/InsightProject.git
#sets the new remote

git remote -v
#verifies the new remote

git push -u origin master
# Pushes the changes in your local repository up to the remote repository you specified as the origin
