const express = require("express");
const app = express();
const path = require("path");
const session = require("express-session");
const MongoStore = require("connect-mongo");
const fileUpload = require("express-fileupload");
require("dotenv").config(); // Load environment variables from .env file

const secret = process.env.SESSION_SECRET;
const port = 3000 || process.env.PORT;

app.set("views", path.join(__dirname, "src", "views")); //Setting the path of views to src/views
app.set("view engine", "ejs");

app.use(express.static(path.join(__dirname, "src", "public"))); //For static files like CSS and client-side JS

// Middlewares

// Session middleware
app.use(
  session({
    secret: secret,
    resave: false,
    saveUninitialized: true,
    store: MongoStore.create({
      mongoUrl: "mongodb://localhost:27017/lekhapadi",
    }),
    cookie: { maxAge: 180 * 60 * 1000 }, // 3 hours
  })
);

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Middleware to handle file uploads
app.use(fileUpload());

const routes = require("./src/routes");
app.use("/", routes);

// Middleware for handling 404 errors
app.use(function (req, res, next) {
  res.status(404).render("pageNotFound");
});

// Middleware for handling 500 errors
app.use(function (req, res, next) {
  res.status(500).render("serverError");
});

app.listen(port, () => {
  console.log(`server is running at port ${port}`);
});
