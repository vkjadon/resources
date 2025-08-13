# Folder Structure
## Add two folders in `scr` folder
* `components` folder
* `pages` folder

## Add three folders in the `pages` folder below
    
* `Home` folder
    * Create `Home.jsx` and `Home.css`
    * Add `className='home'` in the return div of `Home.jsx`
    * Import `Home.css` into `Home.jsx`

* `Login` folder
    * Create `Login.jsx` and `Login.css`
    * Add `className='login'` in the return div of `Login.jsx`
    * Import `Login.css` into `Login.jsx`
    
* `Player` folder
    * Create `Player.jsx` and `Player.css`
    * Add `className='player'` in the return div of `Player.jsx`
    * Import `Player.css` into `Player.jsx`

## Add three folders in the `components` folder as below
    
* `Navbar` folder
    * Create `Navbar.jsx` and `Navbar.css`
    * Add `className='navbar'` in the return div of `Navbar.jsx`
    * Import `Navbar.css` into `Navbar.jsx`

* `TitleCards` folder
    * Create `TitleCards.jsx` and `TitleCards.css`
    * Add `className='titlecards'` in the return div of `TitleCards.jsx`
    * Import `TitleCards.css` into `TitleCards.jsx`

* `Footer` folder
    * Create `Footer.jsx` and `Footer.css`
    * Add `className='footer'` in the return div of of `Footer.jsx`
    * Import `Footer.css` into `Footer.jsx`

## Mount Home Page and Navbar Component

* `Home` Page
    * Import `Home` in `App.jsx`
    * Mount `<Home />` in return div of `App.jsx`
    * Mount `Navbar` in `Home.jsx` by importing and adding `<Navbar/>` in return div of `Home.jsx`

* Design `Navbar`
    * Create two divisions with class `navbar-left` and `navbar-right` for navbar links
    * Add logo image after importing logo and add list elements in the left section
    * Add icons and links in the right section

# Updating Navbar.css

* Add css to `.navbar` class

```css 
.navbar { 
    width: 100%; 
    padding: 20px 6%; 
    display: flex; 
    justify-content: space-between; 
    position: fixed; 
    font-size: 14px; 
    color: #e5e5e5; 
    background-image: linear-gradient(180deg, rgba(0,0,0,0.7) 10%, transparent); 
    z-index: 1; 
    }
```

It styles a fixed-width, top-positioned navigation bar (.navbar) that spans the full width of the screen, uses a horizontal flex layout to space its child elements, has internal padding, a small font size, light text color, a dark-to-transparent top-down gradient background, and is layered above other content using z-index.

* Add css to `.navbar-left` class

```css
.navbar-left {
    display: flex;
    align-items: center;
    gap: 50px;
}
```

It creates a horizontal flex container (.navbar-left) where items are vertically centered and spaced 50px apart.

```css
.navbar-left img {
    width: 90px;
}
```

It sets the width of all <img> elements inside .navbar-left to 90 pixels.

```css
.navbar-left ul {
    display: flex;
    list-style: none;
    gap: 20px;
}
```

It styles the <ul> inside .navbar-left as a horizontal flex container with no bullet points and 20px spacing between list items.

```css
.navbar-left ul li{
    cursor: pointer;
}
```

It makes each list item (li) inside .navbar-left ul show a pointer cursor on hover, indicating they are clickable.


