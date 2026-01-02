# Notes

- This document contains notes about what path we took, and why.
- This is not necessarily the *same* path I took ... but a linearlized, non-convoluted version of my thoughts at the moment meant to be an easier read.


STEP 1:

I found Strudel.CC and was amazed ... you could play music on your machine without knowing *anything* about it.
Given that I knew nothing about it, it was hard for me to come up w/ good strudel code.

So I asked Claude to 'vibe' code me some audio examples for a podcast intro ... and voila it did!

It was really, really good. Only problem? Not all outputs were valid ... and sometimes coding errors would go uncaught ...

So I started wondering ...

"Could I somehow create music that matches my mood? Where my mood is inferred from my computer activity? And entirely locally on *my* machine?"

STEP 2:

Asked myself a question ... "Can we formulaicly produce music? Why? Why not?"

Claude helped me come up w/ V1 schema in this chat - constrain being "Any combination shouldn't possibly be discordant - all harmonious regardless of what combination I use":
https://claude.ai/chat/4d588443-93b6-4859-8ccb-5f3c8c4e9a97

I was able to generate some audio .. then get another LLM (Gemini) to classify the audio config to specific themes ... mostly accurately. But there were some issues.

Then I asked, "Wow ... this is great, but it's too hard to 'discern' ... impossible to know what's 8-bit music, Indian wedding, xmas, funeral clown, etc. for a normal human ear"

I asked Claude to give me more params ... ones that don't sound discordant regardless of the combination I choose. This was much harder ...
Then Claude helped me come up with V2 params ... however, this time, you *could* end up shooting yourself w/ a footgun.

Best way to *minimize* risk of problematic combinations was to use ENUMS to re-use some good combintations to ensure output audio *largely* consistently decent ...

---

Beyond just this, I was able to recruit Claude's help in creating a `synth` that uses numpy, sklearn, and scipy to create 'music' entirely using Python (... w/ C/C++ under the hood ... ofc)
It's not perfect ... but I tweaked it with the help of ChatGPT Pro and Gemini Pro .. and it's in a good condition right now.

I tried using Pyo and Pedalboard, but turns out I didn't really need them. 

STEP 3:

I liked the output ... however, it had issues w/ creating certain specific combinations - such as "Egyptian Summer", "Arabian Nights", "Indian Culture" ... the thing about those is that all of them
have very stereotypical beats that the existing system couldn't replicate.

So I planned on extending V3 to include more melody presets ... and not just that - but allow it to specify custom melody sequences.
This is still a plan ... why?
- Because I wasn't interested in having to add various culture specific melodies as 'pre-sets' ... that'd make the config incredibly complex.
- Allowing the input to be a list of chords that could be played could be a footgun - the person who writes the config now truly needs to understand *what* perhaps "Egyptian Summer"-ish melody sounds like.
I wasn't sure it'd be an easy task for someone entirely untrained in music to do - and definitely not a tiny LLM (I was looking at <500 mb models>) - not without fine-tuning atleast, could do without
butchering it every once in a while.

So now I started exploring some other avenues ... perhaps I'm overcomplicating it and there's better ways.

- RAVE: You can essentially explore a latent space of audio ... and invert it ... and supposedly get good audio out. That wasn't the case with me. The examples I tried mostly sounded just ... not great.
And it wasn't computationally cheap either. 

Also, how'd I first map an input 'vibe' to the same latent space that RAVE maintains?

There were ... some ways:

- You could download dataset of audio that's labelled w/ text descriptions. This exists, btw. For example:
  Then get CLAP to give you the text *and* audio embeddings *and* RAVE embeddings ... do RAG over them for an existing 'vibe' ... take a weighted average of the RAVE embedding and invert it.

  However, .. I'd need a well-trained RAVE dataset which doesn't exist ... and I couldn't be bothered to rent GPUs over at vast.ai to train this for a few days ...

  Not to mention the problems w/ merely taking a RAG to 'estimate' the target song in RAVE latent space ... that's a deep hole. What if you search for the word "night" and the only song
  that exists is "Arabian nights"? And soon your night audio sounds like you are in the desert? 

  It's need a dataset of ton of 'neutral' and 'orthogonal' concepts ... that could be independently combined. I'm not and was not certain if this is/was possible. I didn't pursue this further.

  Not to mention the model isn't exactly small. But some variations of RAVE were quite small from 50mb to 150 mb ... but unless you want to hear orca or bird noises I won't bother lmao.

- I could use something like MusicVAE and Magenta ... however this was a nightmare. The project is essentially abandonware at this point and I had to just give up.
  Additionally ... same problem exists with MusicVAE - how do I convert a vibe like "Night" to the latent space and then invert it back to music? So I gave up.

  Not to mention the model isn't exactly small.

- Then came MidiLLM - this music essentially doesn't sound harmonius. I abandoned this pretty quickly ... not to mention the LLM size was not exactly tiny ... so more compute and bad results? nah thanks.

- I could use a massive LLM running locally ... but that'd just defeat the point of a 'light' application running in the background all the time.
  Not to mention even large LLMs didn't do well at composing music! Only SOTA models could do *well*

  I also realized that, for this reason I should probably only make the app compatible w/ Apple Silicon macs ... old Macs are too slow and are going to not be able to handle the continuous load.
  My goal then became to run something that could "run easily on even an M1 macbook air!"

- After all these attempts ... I just gave up lol. And returned to my original approach of using a custom fine-tuned LLM to drive the V2 config ...

STEP 5:

Now ... how tf do I generate the configs ... all locally? Dumb LLMs are too dumb - tried few shot - tried 'many' shot (tons of examples) ... didn't work. Not to mention generating 
structured output w/ a tiny LLM is always a pain - they tend to run on (read here ...) - so I applied a frequency penalty to get the LLM to not talk too much.

On top of that ... I generated ~2-4k examples using Claude (best one I found for generating music) ... did best of N at a non-zero temperature to pick the best output ... 
and then used it as a source of truth. I 'vibe' tested it by manually playing a lot of those tracks and they sound good.

Then I used that data to fine-tune a small model to generate configs ... and it works *decently well* at generalization across genres and moods!
I actually used this dataset of 'map of music' to create the original Claude dataset! and for each genre even generated multiple alternatives!

And now I have a model that can create music locally!!!

STEP 6:

Now I'm thinking ... "hmm ... running an LLM - even a small one - would be a drain. What if I want to reduce the compute needs *even further*?"

So I created a dataset of rather 'pure' concepts - water, night, morning, cool .. etc. ... all using Claude btw ... some ~1-2k concepts like this ... and allow interpolation between them.

Wrote a custom 'tweening' fn that takes N configs, and N weights and 'merges' them to a new config that averages the continuous values but probabilistically weights discrete bits and combines them
... it's not perfect but does the job.

KEY here was to generate a 'justification' field in Claude, and train the small LLM on it to get it to 'think' before it generates the output. I've done this before and usually results in better results
by 'emulating' a chain of thought - though this is NOT the right way to do this and you should ideally be using GRPO for it ... but this was simple task ...



STEP <FINAL>:

How did I create the UI? It looks retro ... and rather pretty.
I used Textual - a TUI library for Python. Only concern is that it's a TUI library! How do I deploy my own MacOS app running Textual?
I used a Webview, and rendered Textual TUI *inside* it. I also created a MacOS menu bar drop-down that can be used to open/close it - as is the case w/ most MacOS apps today that 'live in the background'.

Because this combination of "TUI but in native MacOS window" is rather unusual I had to fix a ton of tiny things here and there to make sure it's smooth for end-users to use - from tiny UI errors, issues
w/ interoperability of common MacOS bindings, etc. ... but I was able to get it working in a decent state.

I also had to tie and clean up the lifecycle of various processes - the Menubar app, the UI for the app, the Python server actually running it, the code that generates audio ... etc.
It was definitely a bit of a hassle.

Then had to create UIs w/ various knobs.

Also had to figure out what permissions to get to track keyboard and mouse speed, currently window title, text being typed. This was Mac specific.

Then how to bundle everything up together into a single binary using PyInstaller ... fun stuff.