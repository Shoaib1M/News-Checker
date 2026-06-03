/*
FILE PURPOSE:
Defines the Mongoose schema and model for a User.
This tells MongoDB exactly what data fields a User document should have, what data types they are, and any rules (like required or unique).

FLOW:
1. Define the shape of the data (the schema).
2. Create a Mongoose model from that schema.
3. Export the model so other parts of the app (like auth.js routes) can query or save users to the database.

USED BY:
- server/routes/auth.js (to create or find users during login)
- server/models/Check.js (to link a check to a specific user)
*/

import mongoose from "mongoose";

// Create a new Mongoose Schema. A schema represents the structure of a particular document in MongoDB.
const userSchema = new mongoose.Schema(
  {
    // googleId: We use Google OAuth for login, so we store the unique ID Google gives us.
    googleId: {
      type: String,
      required: true, // A user cannot exist without a Google ID in our app
      unique: true,   // No two users can have the same Google ID
      index: true,    // Adding an index makes searching for a user by googleId much faster
    },
    
    // email: The user's email address from Google.
    email: {
      type: String,
      required: true,
      unique: true,
    },
    
    // name: The user's display name.
    name: {
      type: String,
      required: true,
    },
    
    // avatar: A URL pointing to the user's profile picture.
    avatar: {
      type: String,
      default: "", // If Google doesn't provide one, default to an empty string
    },
  },
  // Options object
  // timestamps: true automatically adds 'createdAt' and 'updatedAt' fields to the document.
  { timestamps: true }
);

// Export the model. 
// "User" is the name of the model, which means Mongoose will automatically look for a MongoDB collection named "users" (lowercased and pluralized).
export default mongoose.model("User", userSchema);
