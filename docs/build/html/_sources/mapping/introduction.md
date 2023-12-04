# Introduction to interactive maps

The embedding plot we just saw looks cool, but unless we can make it explorable,
it remains just a cloud of dots.



The internet allows us to display results in an interactive format.

To understand the possibilities here, it's worth a quick recap on how the internet works

- A human with a browser requests a url
- The browser sends a message to a server at the specified address
- The server receives the request, and sends a response
- The browser receives the response and builds or amends a webpage.

Our browsers can receive instructions in the format of `html` or `javascript`.

Html is a markup language, it tells the browser to put a box here containing this text, and a box there containing this image.

With javascript we can manipulate the webpage, and define how it should behave based on how we interact with it.

## Server side and browser side code

If we want interactivity in our plots, we can achieve this in two ways.

Let's take the example of a filter.

- Browser-side interactivity

When we click on a button, the browser has already received instructions on how to filter data, and does so.

- Server-side interactivity

When we click on a button, we send a message to the server that says "please filter the data", the server filters the data and sends it back to us.

### Browser-only apps.

An interactive app map that runs entirely on the browser has the advantage of being very easy to share. Since we don't need a server, we can just host it on github pages ([example](https://mcallaghan.github.io/interactive-impacts-map/), [source code](https://github.com/mcallaghan/interactive-impacts-map).

Doing this means we have to write a lot of javascript code. There are some higher and lower level frameworks available [Vega](https://vega.github.io/vega/), [d3.js](https://d3js.org/). But things can get quite tricky the more complex our requirements are.

### Server side apps using frameworks

Libraries like Shiny and Dash help us to write a lot of the code - particularly the data manipulation work - in a language we might be more used to as data scientists. Because a browser doesn't know how to run R or Python, this code has to run on a server. If we want to share this with others we will need our own server, or for somebody to host the app.

