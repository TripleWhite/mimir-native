#!/usr/bin/env node
/**
 * Chrome Extension 安装前验证
 * 检查 manifest 和文件结构
 */

const fs = require('fs');
const path = require('path');

console.log('🔍 Chrome Extension 验证\n');

let errors = [];
let warnings = [];

// 1. 检查 manifest.json
console.log('1. 检查 manifest.json...');
try {
    const manifest = JSON.parse(fs.readFileSync('manifest.json', 'utf8'));
    
    // 必需字段
    const required = ['manifest_version', 'name', 'version', 'permissions'];
    for (const field of required) {
        if (!manifest[field]) {
            errors.push(`缺少必需字段: ${field}`);
        }
    }
    
    // 检查 manifest_version
    if (manifest.manifest_version !== 3) {
        warnings.push(`建议使用 Manifest V3 (当前: ${manifest.manifest_version})`);
    }
    
    // 检查 host_permissions
    const hasClaude = manifest.host_permissions?.some(p => p.includes('claude.ai'));
    const hasChatGPT = manifest.host_permissions?.some(p => p.includes('chatgpt.com'));
    if (!hasClaude) {
        warnings.push('host_permissions 缺少 claude.ai');
    }
    if (!hasChatGPT) {
        warnings.push('host_permissions 缺少 chatgpt.com');
    }
    
    console.log('   ✅ manifest.json 格式正确');
    console.log(`   名称: ${manifest.name}`);
    console.log(`   版本: ${manifest.version}`);
    console.log(`   权限: ${manifest.permissions?.join(', ') || '无'}`);
    
} catch (e) {
    errors.push(`manifest.json 解析错误: ${e.message}`);
}

// 2. 检查必需文件
console.log('\n2. 检查必需文件...');
const requiredFiles = [
    'manifest.json',
    'background.js',
    'content.js',
    'popup.html',
    'popup.js',
    'icons/icon16.png',
    'icons/icon48.png',
    'icons/icon128.png'
];

for (const file of requiredFiles) {
    if (fs.existsSync(file)) {
        const stats = fs.statSync(file);
        console.log(`   ✅ ${file} (${(stats.size / 1024).toFixed(1)} KB)`);
    } else {
        errors.push(`缺少文件: ${file}`);
    }
}

// 3. 检查 JS 语法
console.log('\n3. 检查 JS 文件语法...');
const jsFiles = ['background.js', 'content.js', 'popup.js', 'inject.js'];
for (const file of jsFiles) {
    if (fs.existsSync(file)) {
        try {
            const content = fs.readFileSync(file, 'utf8');
            // 简单检查：是否能解析为 JS
            new Function(content);
            console.log(`   ✅ ${file} 语法正确`);
        } catch (e) {
            errors.push(`${file} 语法错误: ${e.message.split('\n')[0]}`);
        }
    }
}

// 4. 检查图标
console.log('\n4. 检查图标...');
const iconSizes = [16, 48, 128];
for (const size of iconSizes) {
    const iconPath = `icons/icon${size}.png`;
    if (fs.existsSync(iconPath)) {
        console.log(`   ✅ icon${size}.png`);
    } else {
        warnings.push(`缺少图标: icon${size}.png`);
    }
}

// 5. 总结
console.log('\n' + '='.repeat(60));
if (errors.length === 0) {
    console.log('🎉 验证通过！可以安装到 Chrome');
    console.log('\n安装步骤:');
    console.log('1. 打开 Chrome，访问 chrome://extensions/');
    console.log('2. 开启右上角"开发者模式"');
    console.log('3. 点击"加载已解压的扩展程序"');
    console.log('4. 选择此文件夹');
    console.log('5. 打开 Claude 或 ChatGPT 页面测试');
} else {
    console.log(`❌ 验证失败 (${errors.length} 个错误)`);
    errors.forEach(e => console.log(`   - ${e}`));
}

if (warnings.length > 0) {
    console.log(`\n⚠️ 警告 (${warnings.length} 个)`);
    warnings.forEach(w => console.log(`   - ${w}`));
}
console.log('='.repeat(60));

// 生成安装指南
if (errors.length === 0) {
    console.log('\n📝 快速测试清单:');
    console.log('□ 打开 chrome://extensions/');
    console.log('□ 加载扩展');
    console.log('□ 打开 claude.ai');
    console.log('□ 按 F12 打开控制台');
    console.log('□ 确认看到 "[Mimir] Extension Loaded on claude"');
    console.log('□ 点击扩展图标，测试连接');
}

process.exit(errors.length > 0 ? 1 : 0);
