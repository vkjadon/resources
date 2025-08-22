## About Tic-Tac-Toe

Tic-Tac-Toe is a simple two-player game usually played on a 3×3 grid. Players take turns marking one empty square with `X` or `O`. Player who takes first turn is assigned `X`. The goal is to be the first to get three of your symbols in a row — horizontally, vertically, or diagonally. The game is drawn if all 9 squares are filled without anyone making 3 in a row.

## Setting up the Project

```js
npm create vite@latest
```

    - project name : p2
    - framework : react
    - variant : javascript

```js
cd p2
npm install
npm run dev
```

The react app locates `index.html` which has <div id="root"></div> tag and one script tag to load `main.jsx`. The `main.jsx` tells react to render <App /> component into the `root` div. The component <App /> is defined in `App.jsx` which is loaded using `import` in `main.jsx`. In other words, the content of `App.jsx` is what the user actually sees on the screen.

Remove all the content of `App.jsx` and `App.css`

## Create a Square

We need a clickable square where a player will click to take the turn. We can use HTML `<button>` element

```js
import "./App.css";

export default function Square() {
  return <button className="square"></button>;
}
```

Check the browser after adding the following style in `App.css`.

```css
.square {
  background: #ffffff;
  border: 1px solid #999;
  float: left;
  font-size: 24px;
  font-weight: bold;
  line-height: 34px;
  height: 34px;
  margin-right: -1px;
  margin-top: -1px;
  padding: 0;
  text-align: center;
  width: 34px;
}
```

You will see one square button on the screen.

Now, we have to display `X` of `O` in the button on the button click. So, we have to handle click event using `onClick`. Also, the state needs to be updated.

```js
import React, { useState } from "react";

import "./App.css";

export default function Square() {
  const [value, setValue] = useState(null);

  function handleClick() {
    setValue("X");
  }

  return (
    <button className="square" onClick={handleClick}>
      {value}
    </button>
  );
}
```

React function defines a Square component that displays a button that starts empty and changes to X on the first click. To see this in action, check the browser and click in box.

`onClick={handleClick}` tells `react` to call the function `handleClick` when the button is clicked. If we write `onClick={handleClick()}` the function would be called immediately during render instead of waiting for the click.

The function `handleClick` set the value to `X`.

You will see two square in case you add `Square` component in `main.jsx` as below.

```js
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.jsx";
import Square from "./App.jsx";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <>
      <App />
      <Square />
    </>
  </StrictMode>
);
```

It is important to note that you will get the same element when you import `App` or `Square` through `./App.jsx`. Now, the browser will render two square boxes.

This activity was done just reaffirm the concept of component handling in react. `Undo` the changes you have done in the `main.jsx`.

## Creating Board

The board for this game is a `3 x 3` grid. So, let us create another component with name `Board` which will use `<Square />` as component we just created. Now, we will `export default` the `<Board />` component, so, **remove** `export default` from `Square` function.

```js
export default function Board() {
  return (
    <>
      <div className="board-row">
        <Square />
        <Square />
        <Square />
      </div>
      <div className="board-row">
        <Square />
        <Square />
        <Square />
      </div>
      <div className="board-row">
        <Square />
        <Square />
        <Square />
      </div>
    </>
  );
}
```

To see this in action, check the browser and click in box. You find the boxes are in one row only. To make it a `3 x 3` grid, add the following css in the `App.css`.

```css
.board-row:after {
  clear: both;
  content: "";
  display: table;
}
```

## Let Board Knows the State of Each Square

Currently, each Square component maintains a part of the game’s state. To check for a winner in a tic-tac-toe game, the Board would need to know the state of each of the 9 Square components.

So, the state variable are to be in the scope of the `Board`. In case we shift that in `Board`, we have to handle the button click inside board and pass the `value` and the `handleClick` function to `Square` function as props. So, we can use:

```js
export default function Board() {
  const [value, setValue] = useState(null);
  function handleClick() {
    setValue("X");
  }
return (
  <>
    <div className="board-row">
      <Square value = { value } onSquareClick = {() => handleClick()} />
   ....
   ....
```

and the Square function is update as

```js
function Square({ value, onSquareClick }) {
  return (
    <button className="square" onClick={onSquareClick}>
      {value}
    </button>
  );
}
```

If make changes in all the `<Square />` components, then all the Squares will have `X` when clicked in any of the Square. Our aim is to update with the individual values associated with the square.
Let us store these values in an array. So

```js
const [squares, setSquares] = useState(Array(9).fill(null));
```

Now, `handleClick` to know the index of the square to update the square text

```js
import React, { useState } from "react";

import "./App.css";

function Square({ value, onSquareClick }) {
  return (
    <button className="square" onClick={onSquareClick}>
      {value}
    </button>
  );
}

export default function Board() {
  const [squares, setSquares] = useState(Array(9).fill(null));

  function handleClick(i) {
    const nextSquares = squares.slice();
    nextSquares[i] = "X";
    setSquares(nextSquares);
  }

  return (
    <>
      <div className="board-row">
        <Square value={squares[0]} onSquareClick={() => handleClick(0)} />
        <Square value={squares[1]} onSquareClick={() => handleClick(1)} />
        <Square value={squares[2]} onSquareClick={() => handleClick(2)} />
      </div>
      <div className="board-row">
        <Square value={squares[3]} onSquareClick={() => handleClick(3)} />
        <Square value={squares[4]} onSquareClick={() => handleClick(4)} />
        <Square value={squares[5]} onSquareClick={() => handleClick(5)} />
      </div>
      <div className="board-row">
        <Square value={squares[6]} onSquareClick={() => handleClick(6)} />
        <Square value={squares[7]} onSquareClick={() => handleClick(7)} />
        <Square value={squares[8]} onSquareClick={() => handleClick(8)} />
      </div>
    </>
  );
}
```

The `handleClick` function creates a copy of the `squares` array (`nextSquares`) with the JavaScript slice() Array method. Then, `handleClick` updates the `nextSquares` array to add X to the i<sup>th</sup> index square.

Calling the `setSquares` function lets React know the state of the component has changed. This will trigger a re-render of the components that use the `squares` state of the Board as well as its child components (the Square components that make up the board).

## Taking Turn ('O' or 'X')

To take turn or putting `X` and `O` alternatively, we have to track/monitor this state also. Let us consider the state as boolean as below and toggle this whenever we call `setSquares` using `if` - `else` block. Update the `Board` as below.

```js
export default function Board() {
  const [squares, setSquares] = useState(Array(9).fill(null));
  const [xIsNext, setXIsNext] = useState(true);

  function handleClick(i) {
    const nextSquares = squares.slice();
    if (xIsNext) {
      nextSquares[i] = 'X';
    } else {
      nextSquares[i] = 'O';
    }
    setSquares(nextSquares);
    setXIsNext(!xIsNext);
  }
  return (
```

Check the browser, you will find that when we click of the different squares, it shows `X` and `O` alternatively. But, the problem now is when you click the already filled square it updates the value continuously. We need to fix this. This we can do by checking the state of the square, and if it is not null, it should not execute `setSquare`. Update `handleClick(i)` as below:

```js
function handleClick(i) {
  const nextSquares = squares.slice();
  if (squares[i]) return;  //Add this line
  ...
  ...
}
```
## Check Winner

Let us create a function to check winner. There are nine possible ways to win a board. 

```js
function calculateWinner(squares) {
  const lines = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6],
  ];
  for (let i = 0; i < lines.length; i++) {
    const [a, b, c] = lines[i];
    if (squares[a] && squares[a] === squares[b] && squares[a] === squares[c]) {
      return squares[a];
    }
  }
  return null;
}
```

The `index` values are stored in an array (`line`). The logic for check takes the first value of the `line` i<sup>th</sup> array, stores it in `a` and then fetch the value of `squares[a]`. The check return `true` if all the three values `squares[a], squares[b]` and `squares[c]` are same. 

The function either returns `X` or `O` if winner exists and `null` if no winner exists. Now, you can call this function in the `Board` component just before `return` block.

```js
const winner = calculateWinner(squares);
return (
    <>
    ..
    </>
)
```
The returned value will be stored in the `winner`, its is `X` or `O` or `null`. It is important to note that this function is executed as and when React render or re-render the `Board` component whenever any of the state associated with `Board` is updated.

It’s not “automatically” executed in the sense that JavaScript is watching squares and calling it like a magic trigger. What actually happens is pure React re-rendering logic

- You click a square -> `handleClick(i)` calls `setSquares(nextSquares)`
- `setSquares()` updates state
- In React, updating state tells React that the component’s data has changed, so, re-run this functional component to get the new UI.
- React re-runs Board() -> The whole body of the Board function runs again from the top with the updated squares state.
- When React reaches `const winner = calculateWinner(squares);` it calls `calculateWinner()` again. This time with the new squares array that includes the latest move and returned value is assigned in `winner`.

Add `<div className="status">{ winner } : Wins</div>` to display the winner. 
Check the browser and we get the winner if one of the nine winning line matches the filled pattern. Now, you will notice that you can further click the square and fill it with either `X` or `O`. We want that game should be over once winner exists. No futher move to be allowed. This you can achieve by adding another condition with logical operator in `if (squares[i]) return;`. So replace this with `if (calculateWinner(squares) || squares[i]) return;`. This ensure no futher move in case either of the conditions is satisfied.

Final Code is as below:

```js
import React, { useState } from "react";

import "./App.css";

function Square({ value, onSquareClick }) {
  return (
    <button className="square" onClick={onSquareClick}>
      {value}
    </button>
  );
}

export default function Board() {
  const [squares, setSquares] = useState(Array(9).fill(null));
  const [xIsNext, setXIsNext] = useState(true);

  function handleClick(i) {
    const nextSquares = squares.slice();
    if ( calculateWinner(squares) || squares[i]) return;
    if (xIsNext) {
      nextSquares[i] = "X";
    } else {
      nextSquares[i] = "O";
    }
    setSquares(nextSquares);
    setXIsNext(!xIsNext);
  }
  const winner = calculateWinner(squares);

  return (
    <>
      <div className="winner-state">{ winner } : Wins</div>
      <div className="board-row">
        <Square value={squares[0]} onSquareClick={() => handleClick(0)} />
        <Square value={squares[1]} onSquareClick={() => handleClick(1)} />
        <Square value={squares[2]} onSquareClick={() => handleClick(2)} />
      </div>
      <div className="board-row">
        <Square value={squares[3]} onSquareClick={() => handleClick(3)} />
        <Square value={squares[4]} onSquareClick={() => handleClick(4)} />
        <Square value={squares[5]} onSquareClick={() => handleClick(5)} />
      </div>
      <div className="board-row">
        <Square value={squares[6]} onSquareClick={() => handleClick(6)} />
        <Square value={squares[7]} onSquareClick={() => handleClick(7)} />
        <Square value={squares[8]} onSquareClick={() => handleClick(8)} />
      </div>
      <div className="status"> { winner ? "Game Over" : null } </div>
    </>
  );
}

function calculateWinner(squares) {
  const lines = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6],
  ];
  for (let i = 0; i < lines.length; i++) {
    const [a, b, c] = lines[i];
    if (squares[a] && squares[a] === squares[b] && squares[a] === squares[c]) {
      return squares[a];
    }
  }
  return null;
}
```

`App.css`

```css
.square {
  background: #ffffff;
  border: 1px solid #999;
  float: left;
  font-size: 24px;
  font-weight: bold;
  line-height: 34px;
  height: 34px;
  margin-right: -1px;
  margin-top: -1px;
  padding: 0;
  text-align: center;
  width: 34px;
}

.board-row:after {
  clear: both;
  content: '';
  display: table;
}

.winner-status {
  margin-bottom: 10px;
}

.status {
  margin-bottom: 10px;
  color: brown;
}
```


