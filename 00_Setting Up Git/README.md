# Setting Up Your Local Git Repository

You might be wondering: "How do I make changes to this repo?" - and the short answer is: "You don't!". However, let's look at a pattern you can use to effectively follow along with the course!

### Pre-Requisites

You'll need to make sure a few things are completed before doing this:

- You have set-up your SSH key on GitHub.com. More details can be found [here](https://github.com/AI-Maker-Space/Interactive-Dev-Environment-for-LLM-Development?tab=readme-ov-file#-setting-up-keys-and-tokens)
- If you're on Windows, you'll need to make sure your WSL2 is set-up. More details on that can be found [here](https://github.com/AI-Maker-Space/Interactive-Dev-Environment-for-LLM-Development?tab=readme-ov-file#rocket-lets-get-started)

### Creating a New Repository on GitHub and Cloning It

We're going to create a brand new repository on GitHub.com!

You can follow the guide [here](https://docs.github.com/en/repositories/creating-and-managing-repositories/quickstart-for-repositories) to do that!

Here's an example of the settings you should use:

![image](https://i.imgur.com/WQtlxc5.png)

> NOTE: You can use any name - but the process will be easiest if you make sure `Add a README file` is *deselected*.

Once you've created your new repository. You'll need to capture its `SSH` address.

You'll want to copy this address:

![image](https://i.imgur.com/62QNyfz.png)

> NOTE: Your address will be different - you can safely click the 'Copy' icon at the end of the address bar shown above.

Now you can execute the command:

```bash
git clone PASTE_YOUR_ADDRESS_HERE
```

Then we need to change-directory (`cd`) into our newly cloned repository!

```bash
cd YOUR_REPOSITORY_NAME_HERE
```

> NOTE: If you see a warning message like this: `warning: You appear to have cloned an empty repository.` - that means you've successfully followed all the steps so far correctly!

### Adding the AIE8 Repository as an "Upstream Remote"

Now we're going to add the class repository as an "upstream remote".

First, we'll run this command:

```bash
git remote add upstream git@github.com:AI-Maker-Space/AIE8.git
```

We can verify we were successful by using the command:

```bash
git remote -v
```

You should see an output very similar to this (your origin will be a different address)

```
origin  git@github.com:chris-alexiuk/AIE8.git (fetch)
origin  git@github.com:chris-alexiuk/AIE8.git (push)
upstream        git@github.com:AI-Maker-Space/AIE8.git (fetch)
upstream        git@github.com:AI-Maker-Space/AIE8.git (push)
```

Now, let's look at a diagram of how this works:

![image](https://i.imgur.com/7TA9TIu.png)

As you can see - there are ~4 steps when working with this pattern.

1. We first `git pull upstream main` the changes that were made to the AIMS repository - the one we just added called `upstream`.
2. Then we make changes to that code - and eventually we `git add .` it to our `git` history.
3. Then we commit those changes (with a helpful message) to our `git log`.
4. Then we `git push origin main` those changes to our downstream repisitory - which is called `origin`.

This pattern will allow you to create a public or private repository of your choosing, but still work from our repository!

But before we get to doing this - we'll first need to do our "first pull and push"!

### Our First Pull and Push

This is a long command, but it basically just lets us "sync up" our `git log` between the two different remote repositories.

It's easy enough in practice, we first need to `git pull` the content from the AIE1 remote repository hosted on GitHub.com to our local repository.

```bash
git pull upstream main --allow-unrelated-histories
```

Then, we need to `git push` the new content to our empty remote repostory on GitHub.com.

### Conclusion

Now you're done!!!! :tada:

