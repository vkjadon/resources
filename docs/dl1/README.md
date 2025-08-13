```js
npm create vite@latest
```

    - project name : b1
    - framework : react
    - variant : javascript

```js
cd b1
npm install
npm run dev
```

## How Homepage is loaded

The react app locates `index.html` which has <div id="root"></div> tag and another tag to use `main.jsx`. The `main.jsx` tells react to render <App /> function (called components in react) into that `root` div. The component <App /> is defined in `App.jsx`.

In other words, the content of `App.jsx` is what the user actually sees on the screen. It’s your homepage. You can change this file to customize what shows up.

## Vite + React Homepage

When you first open a Vite + React app in the browser, you'll see a clean starter page with the Vite and React logos at the top. These logos are clickable — if you click on them, they’ll take you to the official websites of Vite and React to learn more. Below that, there’s a button with a counter. Every time you click the button, the number increases — this shows how React can keep track of changes (called “state”) and update the page instantly without reloading. You’ll also see a message like “Edit src/App.jsx and save to test HMR.” This means that if you make any changes in the App.jsx file and save, the browser will automatically update without a full page reload — this feature is called Hot Module Replacement (HMR) and it helps developers see changes quickly as they code.

### Vite and React Logos

* Vite and React Logos. Click on them and you’ll be taken to their websites

* The Counter Button. Each time you click, the number goes up

* Behind the scenes, React is remembering the number using a built-in memory called state.

* Message: “Edit src/App.jsx and save to test HMR”

    * This means: "Go to the file App.jsx, change something, and watch it appear immediately on the page."
    * This magic is thanks to Hot Module Replacement (HMR).


What is visible on the hopepage is what is returned by `App` function in `App.jsx`. Let us see the `return` line by line.

```js
return (
    <>
        ....
        <h1>Vite + React</h1>
        .....
        <p className="read-the-docs">
        Click on the Vite and React logos to learn more
      </p>
    </>
  )
```
The return is wrapped in tags `<>...</>` called fragment. The react components can return only one tag. So, you should enclose everything in fragment.

```js
<div>
    <a href="https://vite.dev" target="_blank">
        <img src={viteLogo} className="logo" alt="Vite logo" />
    </a>
    <a href="https://react.dev" target="_blank">
        <img src={reactLogo} className="logo react" alt="React logo" />
    </a>
</div>
```
The first `div` tag render two logos. JSX is React's way of writing HTML in JavaScript. Anything inside {} is treated as JavaScript. `viteLogo` is just a JavaScript variable holding the path to the vite Logo. These lines load the images (logos) for React and Vite so they can be shown on the page.

The following import statments load the images and save the path into variables.

```js
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
```

**The other `div` tag** 


```js
<div className="card">
    <button onClick={() => setCount((count) => count + 1)}>
        count is {count}
    </button>
    <p>
        Edit <code>src/App.jsx</code> and save to test HMR
    </p>
</div>
```

Next `div` produces a button and displays the current number (`count`). Each time you click it, `setCount` increases the number by 1 and `React` updates the screen. 

The user defined variables `count` and `setCount` holding **state** (`count`) and a function to **update** the state (`setCount`) respectively are part of special React feature called `useState`. 

```js
import { useState } from 'react'
```

Think of it like a small memory box that React uses to remember values, like a score or count.

A message in the `p` tag tells you to try editing this file (App.jsx) to see live updates. This is thanks to Hot Module Replacement (HMR) – no full reload needed.


```js
import './App.css'
```
This loads the styling (colors, sizes, spacing, etc.) to make the page look nice.