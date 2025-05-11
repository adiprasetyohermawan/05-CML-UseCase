function calculateAgeRange(dob) {
  const age = new Date().getFullYear() - dob.getFullYear();
  const ageRanges = [
    "19 - 22",
    "23 - 26",
    "27 - 30",
    "31 - 34",
    "35 - 38",
    "39 - 42",
    "43 - 46",
    "47 - 50",
    "51 - 54",
  ];

  if (age < 19) return "Invalid age";
  const index = Math.min(Math.floor((age - 19) / 4), ageRanges.length - 1);
  return ageRanges[index];
}

// Example usage:
const dob = new Date("1990-01-01"); // Replace with actual date of birth
const ageRange = calculateAgeRange(dob);
console.log(ageRange); // Output based on dob value
