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

Html is just a markup language, it tells the browser to put a box here containing this text, and a box there containing this image.

With javascript we can manipulate the webpage, and define how it should behave based on how we interact with it.

If we want interactivity in our plots, we can achieve this in two ways. Let's take the example of a filter.

- Browser-side interactivity

When we click on a button, the browser has already received instructions on how to react to that button, and amends the plot accordingly.

- Server-side
