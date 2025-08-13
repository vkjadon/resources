<h1>App to Add Two Numbers</h1>

## Project Outline

In this project you will learn to add two numbers. Both the numbers are to be added through two input elements and should output sum after validation on a button click.

## Learning Outcome

- Components
  - a React component is a `JavaScript` function that combines markup, CSS, and JavaScript. It returns you `JSX`.
  - a react components are reusable UI elements for the app.
- useState
  - useState is a `React Hook` that lets functional components create and manage their own state. In simple terms, it allows a component to remember information and update it when needed.
  - Hooks are special functions in React that let you “hook into” features like state and lifecycle methods without writing a class.

## Create a Project

```js
npm create vite@latest
```

    - project name : b2
    - framework : react
    - variant : javascript

```js
cd b2
npm install
npm run dev
```

## File Structure

There are number of ways to approach this project. We will introduce commonly used professional file structure. Its a good practice to keep components group in one folder inside a `components` folder inside `src` folder.

```css
b2/
|-- public
|-- src/
│-- |-- components/
|-- |-- |-- formComponents/
|-- |-- |-- |-- InputField.jsx
|-- |-- |-- |-- SubmitButton.jsx
|-- |-- |-- |-- SumForm.jsx
|-- |-- App.css
|-- |-- App.jsx
|-- |-- index.css
|-- |-- main.jsx
|-- ...
|-- index.html

```

## Clean App.jsx

Open `App.jsx` and remove some of the code to make the file look like as below. Alternatively, you can paste the following code in the `App.jsx`

```js
import "./App.css";
function App() {
  return <></>;
}
export default App;
```

## Update `InputField.jsx`

Create an `InputField` component with classname _input_. Recall that a component is a javascript function returning `jsx`. The parameters of the functions are called props in react. So, you can see three props used in the components. We will mount this component in `SumForm.jsx`.

```js
import "./form.css";
function InputField({ placeholder, value, onChange }) {
  return (
    <input
      className="input"
      type="text"
      placeholder={placeholder}
      value={value}
      onChange={onChange}
    />
  );
}
export default InputField;
```

Add the following css in `form.css` in `formComponents` folder

```css
.input {
  margin: 10px;
  padding: 5px;
  font-size: x-large;
  border-radius: 10px;
}
```

## Update `SumForm.jsx`

Update `SumForm.jsx` to define the `SumForm` component, which uses the `InputField` component for user input. To manage (store and update) the value entered in the input field, we use the `useState` hook. Here, `firstNum` holds the current value of the input field, and `setFirstNum` is the function used to update that value. The `onChange` listens for changes in the input field, and the function `setFirstNum(e.target.value)` updates the `firstNum` state with the new value entered by the user. `e` is the event object automatically passed by React when an event happens. `target` refers to the HTML element that triggered the event. In this case, it's the `input` element. If we enter `3492`, `e.target.value` will be `3492`

```js
import InputField from "./InputField";
import { useState } from "react";
function SumForm() {
  const [firstNum, setFirstNum] = useState("");
  return (
    <div>
      <h1>Add Two Numbers</h1>
      <InputField
        placeholder="First Number"
        value={firstNum}
        onChange={(e) => setFirstNum(e.target.value)}
      />
    </div>
  );
}
export default SumForm;
```
## Update `App.jsx`

Now, we are ready to run this app. Before running, we have to mount `SumForm` component in the `App.jsx`.

```js
import SumForm from './components/formComponents/SumForm'
function App() {
  return (
    <>
      <SumForm />
    </>
  )
}
export default App
```

Run the app using `npm run dev` you can inspect the browser. You can monitor the value entered displayed in the component if you have installed the react extension in the browser.

## Update `SubmitButton.jsx`

```js
function SubmitButton({ onClick, label }) {
  return <button onClick={onClick}>{label}</button>;
}
export default SubmitButton;
```

The `SubmitButton` component requires two props. You should mount this component in `SumForm` below the `InputField` component.

```js
function SumForm() {
  const [firstNum, setFirstNum] = useState('')

  const handleAdd = () => {
    alert("Handled " + firstNum)
  }
  return (
    <div>
      ...
      <InputField...
      <SubmitButton onClick={handleAdd} label="ADD" />
    </div>
  )
}
export default SumForm
```

The button click is handled through a function `handleAdd`. When you run this you will get a pop-up <pre>Handled 5</pre>

## Mount another `InputField`

Add `const [secondNum, setSecondNum] = useState('')` for handling the state of second number and mount the `InputField` for the second number. Use the `secondNum` and `setSecondNum` as prop values/functions. You need to update only the `SumForm.jsx`. Make minor changes in the `handleAdd` function to show `secondNumber`. As `sum` also need to be updated and reflected on UI, we have to assign a state using `useState` for this. You can use `[sum, setSum]`.

Now we can update the `handleAdd` to update the `sum` and display it on the document. Add `<p>Sum : {sum} </p>` after the `SubmitButton` component to display. Run the app and test.

```js
const handleAdd = () => {
  setSum(firstNum + secondNum);
};
```

## Handling Wrong Output

1. Now, when we input `3` and `5` in the input fields we get output as `35` instead of `8`. This is because the type of `firstNum` and `secondNum` is `string`. We have to parse these into float using `parseFloat()` method before updating. So, we can update the `handleAdd` as below:

```js
const handleAdd = () => {
  setSum(parseFloat(firstNum) + parseFloat(secondNum));
};
```

2. This works fine till you enter a valid number may be integer or float. But, when we enter some other character, we get `NaN`. So, we have to add some validation that the user has entered a valid number for the summation. The good practice is to display an error message. You can take a state to track the error as well using `useState`. You can use `[error, setError] = useState('')` for this.

## Handling Invalid Numbers

As we want the input you entered is only a number, else it should throw an error. This you can achieve through a function which return `true` when the entered values are valid numbers, else it return `false`. We will use `isNaN` (Not a Number) method for this validation. 

```js
function isValidNumber(val) {
  if (!isNaN(val)) return true;
  else false;
}

const handleAdd = () => {
  if (
    isValidNumber(firstNum) &&
    isValidNumber(secondNum)
  )
    setSum(parseFloat(firstNum) + parseFloat(secondNum));
  else setSum("Error! Enter a Valid Number");
};
```
The `isValNumber` function can be converetd into a clear one line syntax as below

```js
const isValidNumber = (val) => !isNaN(val);
```
In this `isValidNumber` is declared as a constant variable that will hold an arrow function. The arrow function takes `val` to check and returns (explicit return as no curly braces are used) `true` in case `val` is a number (`isNaN(4)` returns `false` but `!isNaN(4)` return `true`)

## Try Yourself

1. Display `Error` in `RED` color.
2. Display the error number specific. If first number is not valid, it should write `First Number is Not Valid` and similarly for second number.
3. Add style to the page