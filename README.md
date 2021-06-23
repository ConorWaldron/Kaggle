# Kaggle

For Kaggle Problems

## General Git commands

- git pull # gets most up to date version of code
- git push # pushes your changes to common git repo online

Flow to do some work:

- Make sure you are on the main branch: `git checkout main`
- Pull down the latest work from the remote: `git pull`
- Checkout a new branch locally: `git checkout -b <branch_name>` (this will create a new branch and will switch you to it locally)
- Do some work...
- Check the status (if you want): `git status`
- Stage your changes: `git add .` (the `.` says stage all files that have changes, you can stage specific files if you only want to add a subset)
- Commit you changes: `git commit -m "some message saying what you have done"`
  - At this point you have a new branch with a commit on it locally, but nothing has been updated on the remote
- Push your changes up to the remote: `git push`
- Create a pull request
