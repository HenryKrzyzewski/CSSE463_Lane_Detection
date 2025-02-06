const fs = require('fs');
const path = require('path');

const dirpath = path.join(__dirname, 'dataset/CarlaData2Fixed/FixData');
const outputPath = path.join(__dirname, 'dataset/split');
const pattern = /^000_([a-zA-Z_]+)_[0-9]{4}\.png$/i;

const trainSplit = 0.7;
const testSplit = 0.2;

// Stolen from StackOverflow
// https://stackoverflow.com/questions/2450954/how-to-randomize-shuffle-a-javascript-array
function shuffle(array) {
    let currentIndex = array.length;
  
    // While there remain elements to shuffle...
    while (currentIndex != 0) {
  
      // Pick a remaining element...
      let randomIndex = Math.floor(Math.random() * currentIndex);
      currentIndex--;
  
      // And swap it with the current element.
      [array[currentIndex], array[randomIndex]] = [
        array[randomIndex], array[currentIndex]];
    }
}

fs.rm(outputPath, { recursive: true }, (err) => {
    if (err) {
        console.log('Error when trying to delete output path');
    } else {
        console.log('Successfully deleted output path')
    }
});

fs.readdir(dirpath, (err, files) => {
    if (err) {
        console.error('Error reading directory: ', err);
    }

    const categories = {};

    files.forEach(file => {
        const match = file.match(pattern);
        if (match) {
            const cat = match[1];
            if (!categories[cat]) {
                categories[cat] = [];
            }
            categories[cat].push(file);
        }
    });

    const categoryNames = Object.keys(categories);
    categoryNames.forEach((name, idx) => {
        const numberInThisCategory = categories[name].length;
        console.log(`Number in ${name}: ${numberInThisCategory}`);
        const trainingImageCount = Math.floor(trainSplit * numberInThisCategory);
        const testImageCount = Math.floor(testSplit * numberInThisCategory);
        const validationImageCount = numberInThisCategory - trainingImageCount - testImageCount;
        console.log(`\tTrain images: ${trainingImageCount}\n\tTest images: ${testImageCount}` + 
            `\n\tValidation images: ${validationImageCount}`);

        const images = categories[name];
        shuffle(images);
        for (let i = 0; i < trainingImageCount; i++) {
            const setPath = path.join(outputPath, 'train');
            const outputFolderName = path.join(outputPath, 'train');
            if (!fs.existsSync(setPath)) {
                fs.mkdirSync(setPath, { recursive: true });
            }
            const fileName = images[i];
            const nameWithoutExtension = path.parse(fileName).name;
            const imgName = nameWithoutExtension + '.jpg';
            
            const sourceMask = path.join(dirpath, fileName);
            const destMask = path.join(outputFolderName, fileName);
            const sourceImg = path.join(dirpath, imgName);
            const destImg = path.join(outputFolderName, imgName);
            fs.copyFileSync(sourceMask, destMask);
            if (fs.existsSync(sourceImg)) {
                fs.copyFileSync(sourceImg, destImg);
            }
        }
        console.log(`Category ${name} train images done`);
        for (let i = trainingImageCount; i < trainingImageCount + testImageCount; i++) {
            const setPath = path.join(outputPath, 'test');
            const outputFolderName = path.join(outputPath, 'test');
            if (!fs.existsSync(setPath)) {
                fs.mkdirSync(setPath, { recursive: true });
            }
            const fileName = images[i];
            const nameWithoutExtension = path.parse(fileName).name;
            const imgName = nameWithoutExtension + '.jpg';
            
            const sourceMask = path.join(dirpath, fileName);
            const destMask = path.join(outputFolderName, fileName);
            const sourceImg = path.join(dirpath, imgName);
            const destImg = path.join(outputFolderName, imgName);
            fs.copyFileSync(sourceMask, destMask);
            if (fs.existsSync(sourceImg)) {
                fs.copyFileSync(sourceImg, destImg);
            }
        }
        console.log(`Category ${name} test images done`);
        for (let i = trainingImageCount + testImageCount; i < numberInThisCategory; i++) {
            const setPath = path.join(outputPath, 'validation');
            const outputFolderName = path.join(outputPath, 'validation');
            if (!fs.existsSync(setPath)) {
                fs.mkdirSync(setPath, { recursive: true });
            }
            const fileName = images[i];
            const nameWithoutExtension = path.parse(fileName).name;
            const imgName = nameWithoutExtension + '.jpg';
            
            const sourceMask = path.join(dirpath, fileName);
            const destMask = path.join(outputFolderName, fileName);
            const sourceImg = path.join(dirpath, imgName);
            const destImg = path.join(outputFolderName, imgName);
            fs.copyFileSync(sourceMask, destMask);
            if (fs.existsSync(sourceImg)) {
                fs.copyFileSync(sourceImg, destImg);
            }
        }
        console.log(`Category ${name} validation images done`);
        console.log(`Category ${name} done (${idx+1}/${categoryNames.length})`);
    });
});
