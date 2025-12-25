import processEvalQueue from '../cron/actions/processEvalQueue';

async function main() {
  try {
    console.log('Invoking processEvalQueue...');
    await processEvalQueue();
    console.log('Done.');
  } catch (e) {
    console.error('Error running processEvalQueue:', e);
    process.exit(1);
  }
}

main();
