const fs = require('fs');
const path = require('path');

const trainPath = path.join(__dirname, 'dataset/split/train');
const testPath = path.join(__dirname, 'dataset/split/test');
const validationPath = path.join(__dirname, 'dataset/split/validation');
const pattern = /^000_([a-zA-Z_]+)_[0-9]{4}\.png$/i;

const categoryCounts = {};

const handleDir = (err, files) => {
    if (err) {
        console.error('Error reading directory: ', err);
    }

    files.forEach(file => {
        const match = file.match(pattern);
        if (match) {
            const cat = match[1];
            if (!categoryCounts[cat]) {
                categoryCounts[cat] = 0;
            }
            categoryCounts[cat]++;
        }
    });
    console.log(categoryCounts);
    let sum = 0;
    for (let c in categoryCounts) {
        sum += categoryCounts[c];
    }
    console.log(`Sum: ${sum}`);
};

fs.readdir(trainPath, handleDir);
fs.readdir(testPath, handleDir);
fs.readdir(validationPath, handleDir);
