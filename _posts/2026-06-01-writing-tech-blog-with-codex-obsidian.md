---
title: "Writing Technical Blog Posts with Codex and Obsidian"
date: 2026-06-01
categories: [agentic-coding]
tags: [codex, obsidian, jekyll, writing]
translation_key: writing-tech-blog-with-codex-obsidian
---

I like writing. More precisely, I feel like I need to write regularly. I usually organize my thoughts in a diary, and when I learn something technical, I write it down separately in my notes.

So I have always thought I should create a blog and manage my writing a little more systematically. But whenever I actually set up a blog, I ended up not posting much. Eventually I would delete it, recreate it, and repeat the same cycle.

The reason I wrote regularly but did not upload much was simple. Once a technical post goes online, it starts to feel like a responsibility. What if I write something incorrect? What if the post looks too rough? Is this even worth publishing on a blog? Those thoughts kept following me.

But now, thanks to the progress of AI, I finally felt that it was time to start blogging again.

## Why a Jekyll blog again?

Whenever I thought about a blog, I first thought of a GitHub Pages-based Jekyll blog that I could customize. The problems were that I had to build the blog myself, and managing posts was more tedious than I expected.

But now we are in the agent era. I described the conditions I wanted, and in a single day I was able to create something close to the blog I had imagined. Managing posts has also become much easier because an agent can help with it.

The conditions I wanted were fairly clear.

- It should have category pages
- It should support toggling between Korean and English posts
- It should be deployable with GitHub Pages
- Posts should be managed as Markdown

The blog is still something I am gradually improving. My next goal is to register it with Google Search Console and clean up the SEO-related settings.

## Writing starts in Obsidian

Here is the actual writing process.

In the past, I would have written a draft, searched for and verified details one by one, polished the sentences again, and only then published the final post. It might have taken an entire day. But now that GPT and Codex exist, I no longer need to hold every part of that process by myself.

I use Obsidian and Codex together to write and revise posts, and I think it works well enough that I wanted to share the process here.

First, I create a `blog` folder in Obsidian and add the following subfolders inside it.

```text
blog/
  raw/
  editing/
  published/
```

Each folder has a simple role.

- `raw/`: rough drafts that have not been polished yet.
- `editing/`: posts currently being revised by Codex.
- `published/`: posts that have been finalized and published to the blog.

At first, I freely write the outline and the ideas I want to include in `raw/`. At this stage, writing down thoughts quickly matters more than sentence quality. It is fine if the tone is rough, and it is fine if the paragraph order is not perfect.

Then I add the `blog` folder as a Codex project. Codex copies the post from `raw/` into `editing/` and writes the revised version there. I ask Codex to review the style, typos, structure, Markdown formatting, and content. In other words, I use it as my own editor.

After reviewing the final version in `editing/`, I move it to `published/`. That keeps `editing/` as a workspace that only contains the post currently being revised.

## Defining editing rules with AGENTS.md

The most important file in this process is `AGENTS.md`.

If you add `AGENTS.md` to the `blog/` folder, you can define the rules Codex should follow when editing posts. For example, I wrote principles like these.

```markdown
# AGENTS.md

- Codex acts as a senior editor and editor for blog posts.
- Preserve the core content and intent of the original draft.
- Write in polite, friendly, professional, and simple Korean.
- Do not directly edit files inside raw/ or published/.
- When editing a post in raw/, first copy it into editing/.
- Make all edits only in the copied file inside editing/.
- Normalize image links as Markdown relative links that work in Obsidian.
```

Once these rules are defined, I do not need to repeat the same instructions every time. I can simply ask, "Please edit this post." In particular, making the roles of `raw/`, `editing/`, and `published/` explicit helps reduce the chance of accidentally modifying the original draft.

Everyone runs their blog differently, so the contents of `AGENTS.md` should be adjusted to fit the workflow. The important part is to clearly tell Codex what role it should play and which files it is allowed to modify.

## Uploading to a Jekyll blog

Once the revision is done, it is time to upload the post to the blog.

I also added my personal GitHub Pages blog repository as a Codex project. Then I ask Codex to move the post into the folder where posts are stored, and to adjust the front matter and image paths to match the blog's Jekyll format.

The rough flow looks like this.

1. Review the final post in `editing/`.
2. Move the post into the blog repository's `_posts` folder or the relevant post folder.
3. Clean up the Jekyll front matter to match the blog format.
4. Check that image paths work correctly in the deployed environment.
5. Run a local build and check for problems.
6. If needed, translate and create the English version as well.

It is worth going through your own posting and deployment process in chat at least once. After that, you can ask Codex to "summarize the posting flow we just followed and write it into AGENTS.md," and the process becomes much more stable the next time.

## Publishing feels less burdensome now

Using this method has made publishing feel much less burdensome.

I can quickly write a rough draft in a stream-of-consciousness style, and typos and tone are revised according to the existing rules. Codex can suggest titles, split the post into sections, and clean up the Markdown syntax, which makes the final post much easier to read.

It is also useful that I can ask for verification when I am not fully confident about the content. Of course, the final responsibility still belongs to me as the person publishing the post. But it is much more efficient than struggling alone with every sentence.

## Then did GPT write this post?

So does that mean GPT wrote this post?

I do not think so. When writing this post, I did not ask, "Write a blog post about how to write technical blog posts with Codex and Obsidian." I had a topic I chose, a line of reasoning and context I had thought through, and a claim I wanted to make.

GPT and Codex only made the formal parts of the post easier to read. In my usage, AI is closer to an editor than an author.

I think this is the difference between AI slop and writing that is not. Does the writing contain the author's thoughts and experience? If it satisfies that condition, isn't it reasonable to get help with sentence cleanup, structure, and typo fixes?

I think the same idea can apply to coding. Do you know why the code was designed that way? If you can explain the reason, then even if you received help from an agent, the code is still close to something you made. An agent is not a tool that replaces my thinking. It is a tool that helps me implement the design and intent I already have more quickly and accurately.
