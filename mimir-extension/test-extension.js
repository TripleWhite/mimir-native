// test-extension.js - 验证扩展结构
const fs = require('fs');
const path = require('path');

const extDir = __dirname;
const requiredFiles = [
  'manifest.json',
  'background.js',
  'content.js',
  'inject.js',
  'popup.html',
  'popup.js',
  'icons/icon16.png',
  'icons/icon48.png',
  'icons/icon128.png'
];

console.log('🔍 验证 Mimir Extension 结构...\n');

let allOk = true;
for (const file of requiredFiles) {
  const fullPath = path.join(extDir, file);
  const exists = fs.existsSync(fullPath);
  const status = exists ? '✅' : '❌';
  console.log(`${status} ${file}`);
  if (!exists) allOk = false;
}

console.log('\n📋 Manifest 验证:');
try {
  const manifest = JSON.parse(fs.readFileSync(path.join(extDir, 'manifest.json'), 'utf8'));
  console.log(`  - Manifest Version: ${manifest.manifest_version}`);
  console.log(`  - Name: ${manifest.name}`);
  console.log(`  - Version: ${manifest.version}`);
  console.log(`  - Content Scripts: ${manifest.content_scripts?.length || 0}`);
  console.log(`  - Permissions: ${manifest.permissions?.join(', ')}`);
  
  if (manifest.manifest_version !== 3) {
    console.log('  ❌ Manifest version should be 3');
    allOk = false;
  } else {
    console.log('  ✅ Manifest V3 confirmed');
  }
} catch (e) {
  console.log(`  ❌ Failed to parse manifest: ${e.message}`);
  allOk = false;
}

console.log('\n' + (allOk ? '✅ 所有检查通过！扩展已准备就绪。' : '❌ 存在错误，请修复。'));
console.log('\n📖 安装指南:');
console.log('1. 打开 Chrome，访问 chrome://extensions/');
console.log('2. 开启右上角"开发者模式"');
console.log('3. 点击"加载已解压的扩展程序"');
console.log('4. 选择:', extDir);
