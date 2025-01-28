CREATE TABLE Users (
    userID INTEGER PRIMARY KEY AUTOINCREMENT,
    fullname TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    isVerified INTEGER DEFAULT 0, -- 0 for false, 1 for true
    verificationTimestamp TIMESTAMP DEFAULT NULL
);

CREATE TABLE UserLogin (
    loginID INTEGER PRIMARY KEY AUTOINCREMENT,
    userID INTEGER NOT NULL,
    salt TEXT NOT NULL,
    username TEXT NOT NULL,
    password TEXT NOT NULL,
    FOREIGN KEY (userID) REFERENCES Users(userID)
); 

CREATE TABLE Blogs (
    blogID INTEGER PRIMARY KEY AUTOINCREMENT,
    userID INTEGER NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    coverImage BLOB, 
    categoryID INTEGER NOT NULL,
    FOREIGN KEY (userID) REFERENCES Users(userID),
    FOREIGN KEY (categoryID) REFERENCES Categories(categoryID)
);

CREATE TABLE Categories (
    categoryID INTEGER PRIMARY KEY AUTOINCREMENT,
    categoryName TEXT NOT NULL
);

CREATE TABLE Comments (
    commentID INTEGER PRIMARY KEY AUTOINCREMENT,
    userID INTEGER NOT NULL,
    blogID INTEGER NOT NULL,
    commentText TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (userID) REFERENCES Users(userID),
    FOREIGN KEY (blogID) REFERENCES Blogs(blogID)
);

CREATE TABLE Likes (
    likeID INTEGER PRIMARY KEY AUTOINCREMENT,
    userID INTEGER NOT NULL,
    blogID INTEGER NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (userID) REFERENCES Users(userID),
    FOREIGN KEY (blogID) REFERENCES Blogs(blogID)
);
