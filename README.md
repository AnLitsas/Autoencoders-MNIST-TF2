# Configure GitHub account - Ubuntu

## Create ssh key
Creating ssh key will create a connection between your device and your Git account.
Follow this path
`cd  ~/.ssh 
`
and create your ssh key with the following command:
```
ssh-keygen -t ed25519 -C "your_email@example.com"
```

Then the following message will appear:
> "Enter file in which to save the key: " 


Give a name id to create both **public** and **private keys** with names whatever you type
e.g. "Tasos_id" will produce: Tasos_id, Tasos_id.pub
	then you will be asked to create and enter your passphrase 2 times. 

## Connect your device with your account 
1)  `cat your_id.pub` 
  
2) Copy the produced key
  
3) Go to github.com >> My account >> settings >> SSH and GPG keys
  
4) Select "New SSH KEY" >> Give a Title (e.g. Home Linux) and paste the key in the section with title "Key" and hit "Add SSH Key"
	
## Clone the repository using the command "git clone" 
Go to the repository of your choice in Git and copy the SSH code.   
e.g. current's repository key is : "git@github.com:AnLitsas/Autoencoders-tf2.git"
   
Create your folder that you want to store the repo, go in there and type
   
```
git clone git@github.com:AnLitsas/Autoencoders-tf2.git
```
Now you have cloned the repository.

## Usefull commands 
```
git checkout
git status
git add 
git commit
git pull
git push

```
### 1) *_Branches_*: This is a way to ***keep safe the main project***. 
Anyone who wants to change something in it, he must create a Branch.
```
git checkout -b branch_name
```
This command will create a branch with the given name. 
To check if you have created the branch, type
```
git status 
```
that will output: 
>"On branch branch_name... 
	No commits yet .... " 

(If you are working with many repositories and if you have created a branch for each one, you can check with the latter command 
	that every time you cd to the repository you want, the branch is automatically changed. )
  

### 2) *_Add Changes_*: If you are ready to upload changes or add something new on Git (for a review), type:
```
git add -A 
```
This, will select every file you have under your current directory and will add it into your branch.
```
git add specific_file.extension 
```
or you can use the above command to add specific files.
	
### 3) *_Commit Changes_*: By commiting, you create a save of your code. That way we can keep track of all changes and switch if we ever need to.
```
git commit -m "message"  
```
(message is probably an ID in order to save)

### 4) *_Push_*: In order to upload your changes and push your branch online you need the command:
```
git push -u origin main 
```
Origin refers to your local branch (u can keep origin or your branch name).
main refers to the "master" branch of the remote repository.

### 5) *_Pull Request_*: In order for the changes to merge with the main branch, someone must create a pull request:

    i) Go on GitHub >> Branches >> find your branch and tap "Create pull request".
    
    ii) There you can comment, add reviewers to get notified for the changes and get feedback.

